"""Detector model: gain/offset, shot noise, read noise, banding, clip,
8-bit quantise.

`apply_detector()` is the last stage of the forward model order (geometry ->
SE yield -> PSF -> pixel-area integration -> **gain/offset -> Poisson ->
Gaussian read -> banding -> clip -> 8-bit quantise**). Takes the noiseless
float SE-yield image from `optics.render()` and everything from here on is
per-capture *independent* (own RNG stream, own dose, own read noise, own
brightness/contrast, own banding realisation -- CLAUDE.md constraint #4).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import CaptureConfig

# ASSUMPTION: CLAUDE.md explicitly excludes "calibrated dose->grey-level"
# (see "Do not implement"). There is no literature-sourced electrons-per-DN
# conversion in scope, so we use a fixed, uncalibrated base gain -- chosen
# only so that the ~0.35-2.5 SE-yield range lands in a usable, mostly
# unsaturated slice of [0, 255] at contrast_gain=1.0. Do not read any
# absolute brightness out of this generator's images as a calibrated
# quantity.
_BASE_DN_PER_YIELD = 100.0

# ASSUMPTION: row banding needs a correlation length along the row axis for
# its autocorrelation to show one (validation test 5); CLAUDE.md gives an
# amplitude (`row_band_amp`) but no length, so this is an engineering
# choice -- see D-016. Expressed in output pixels (rows), not nm: banding is
# a property of the scan/detector electronics, not the physical geometry.
_BAND_CORR_LEN_PX = 15.0


def apply_detector(
    yield_image: np.ndarray, cfg: CaptureConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply the full detector chain to a noiseless SE-yield image.

    `yield_image`: float64 array from `optics.render()` (dimensionless SE
    yield, no units).

    `cfg`: this capture's `CaptureConfig` -- `dose_e_per_px` sets the
    physical electron count Poisson shot noise is drawn at (controls SNR
    only, never brightness -- dose-to-greylevel calibration is explicitly
    out of scope); `read_noise_sigma`/`brightness_offset` are in DN;
    `contrast_gain` is a dimensionless DN-space multiplier;
    `row_band_amp` is DN (0 disables banding); `quantise_8bit` selects the
    return dtype.

    `rng`: this capture's *own* noise RNG stream (independent of the
    geometry-realisation RNG and of the other capture's stream -- see
    `pair.py`). Poisson, read noise, and banding are drawn from it in that
    fixed order for determinism.

    Returns a `(rows, cols)` array: `uint8` in `[0, 255]` if
    `cfg.quantise_8bit`, else `float64` clipped to `[0.0, 255.0]` (used by
    the noiseless-precision validation tests, which need sub-DN precision
    that 8-bit quantisation would destroy).
    """
    # gain: SE yield (dimensionless fraction) -> expected electron count.
    # This is the physical scale Poisson shot noise is drawn at; dose
    # controls SNR only (see module docstring / D-015): higher dose means
    # the Poisson-sampled electron count is *relatively* closer to its
    # mean, not a *brighter* image.
    mean_electrons = yield_image * cfg.dose_e_per_px
    noisy_electrons = rng.poisson(mean_electrons).astype(np.float64)
    noisy_yield = noisy_electrons / cfg.dose_e_per_px

    # offset/gain: noisy yield -> DN, via the fixed uncalibrated base gain
    # plus this capture's independent contrast/brightness settings.
    dn = noisy_yield * _BASE_DN_PER_YIELD * cfg.contrast_gain + cfg.brightness_offset

    # Gaussian read noise, DN, independent per pixel.
    dn = dn + rng.normal(0.0, cfg.read_noise_sigma, size=dn.shape)

    # Row-structured banding: Gaussian-correlated along the row axis (same
    # construction as geometry.py's LER -- white noise, smoothed, rescaled
    # to the target amplitude), broadcast across every column in a row.
    if cfg.row_band_amp > 0:
        n_rows = dn.shape[0]
        raw = rng.standard_normal(n_rows)
        band = gaussian_filter1d(raw, sigma=_BAND_CORR_LEN_PX, mode="reflect")
        band *= cfg.row_band_amp / band.std()
        dn = dn + band[:, np.newaxis]

    dn = np.clip(dn, 0.0, 255.0)

    if cfg.quantise_8bit:
        return np.round(dn).astype(np.uint8)
    return dn
