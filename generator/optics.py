"""Beam optics: Gaussian PSF and exact pixel-area integration.

`render()` is steps 2-4 of the forward model order (geometry -> SE yield ->
**PSF -> pixel-area integration** -> gain/offset -> Poisson -> Gaussian read
-> banding -> clip -> 8-bit quantize): it queries the shared geometry's SE
yield field on a supersampled grid, applies the beam's Gaussian PSF at that
fine grid, then box-averages each k x k block of fine pixels down to one
output pixel -- exact pixel-area integration, not a resize (CLAUDE.md
constraint #2). The two captures of a pair each call `render()`
independently (own origin, own FOV, own PSF sigma, own supersample factor,
own edge_amp_scale) against the *same* GeometryModel, so geometry/LER/cuts/
landmarks are shared while everything optical is independent (constraint
#4).

No noise, gain, or quantisation happens here -- `render()` returns a float64
SE-yield image (dimensionless); see detector.py for what comes next.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import GeometryModel, signed_distance
from .yield_field import se_yield


def render(
    model: GeometryModel,
    origin_nm: tuple[float, float],
    fov_nm: float,
    out_px: int,
    psf_sigma_nm: float,
    k: int,
    edge_amp_scale: float = 1.0,
) -> np.ndarray:
    """Render one noiseless capture of `model` and return a float64
    (out_px, out_px) SE-yield image (dimensionless -- gain/noise/quantise
    are detector.py's job, not this function's).

    `origin_nm`: (x0_nm, y0_nm), the world-coordinate corner of this
    capture's field of view. Output pixel (row, col) is centered at
    world coordinate
    `(x0_nm + (col + 0.5) * pixel_size_nm, y0_nm + (row + 0.5) * pixel_size_nm)`
    -- row increases with y, column increases with x, `pixel_size_nm =
    fov_nm / out_px`. This is the same (x0_nm, y0_nm) that belongs in
    truth.json as e.g. `search_origin_nm`.

    `fov_nm`: physical field-of-view size in nm (square: both axes are the
    same size). `out_px`: output image size in pixels (square).

    `psf_sigma_nm`: beam Gaussian PSF standard deviation, nm.

    `k`: supersample factor. The fine (pre-PSF, pre-integration) grid has
    `out_px * k` samples per axis at fine pixel size
    `fine_step_nm = pixel_size_nm / k`; the PSF is applied at this fine grid
    with `sigma` converted to fine-pixel units
    (`psf_sigma_nm / fine_step_nm`), per CLAUDE.md's forward-model order
    (PSF *before* pixel-area integration).

    `edge_amp_scale`: forwarded to `yield_field.se_yield` -- the
    independent per-capture edge-response amplitude
    (`CaptureConfig.edge_amp_scale`).

    ASSUMPTION: the Gaussian filter uses scipy's default boundary handling
    (`mode="reflect"`) at the edge of the supersampled FOV, with no extra
    margin queried beyond it. This can bias the blurred value within about
    one `psf_sigma_nm` of the FOV's outer edge (a few nm) versus an
    infinite-domain PSF; CLAUDE.md's literal step-6 recipe does not call for
    margin/crop handling, and the region that matters for localisation is
    the FOV interior, not its outermost edge pixels. Revisit if a later
    validation test is sensitive to edge-of-FOV pixels specifically.
    """
    pixel_size_nm = fov_nm / out_px
    fine_step_nm = pixel_size_nm / k
    n_fine = out_px * k

    x0_nm, y0_nm = origin_nm
    fine_idx = np.arange(n_fine)
    xs_nm = x0_nm + (fine_idx + 0.5) * fine_step_nm
    ys_nm = y0_nm + (fine_idx + 0.5) * fine_step_nm
    X_nm, Y_nm = np.meshgrid(xs_nm, ys_nm)  # row ~ y, col ~ x

    d_nm = signed_distance(model, X_nm, Y_nm)
    yield_fine = se_yield(d_nm, model.cfg.fin_height_nm, edge_amp_scale=edge_amp_scale)

    sigma_fine_px = psf_sigma_nm / fine_step_nm
    blurred = gaussian_filter(yield_fine, sigma=sigma_fine_px)

    # Exact pixel-area integration: box-average each k x k block of fine
    # pixels into one output pixel. This is supersampling + area-averaging,
    # not a resize of a finished image (CLAUDE.md constraint #2).
    return blurred.reshape(out_px, k, out_px, k).mean(axis=(1, 3))
