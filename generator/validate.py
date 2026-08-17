"""Validation tests. Tests 1, 6, 7 are the gate: nothing downstream is
meaningful until they pass (CLAUDE.md). All are noiseless (dose set very
high so Poisson shot noise is negligible -- L0-style "dose -> infinity",
not a special no-noise code path) and use `quantise_8bit=False` so
comparisons happen at full float precision, not compressed into 8-bit DN.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from skimage.registration import phase_cross_correlation

from .config import CaptureConfig, GeometryConfig, PairConfig
from .detector import _BAND_CORR_LEN_PX, apply_detector
from .geometry import build_geometry
from .optics import render
from .pair import generate_pair

VALIDATION_DIR = Path(__file__).resolve().parent.parent / "docs" / "validation"

_NOISELESS_DOSE = 1.0e12  # "dose -> infinity": Poisson noise negligible, not skipped

# Search capture psf_sigma_nm used by tests 6/7 (and, pending presets.py,
# the closest thing this repo has to a "production" search default).
# Raised from CLAUDE.md's literal 3.0 nm default to 8.0 nm after a
# boundary-controlled sigma sweep (3/5/8 nm, isolated edge + real intel14
# lattice, small vs 6x-larger FOV) found: (a) zero measurable
# FOV-boundary/reflect-mode contamination at any tested sigma up to 8 nm
# (identical slope/residual at 200px/2000nm vs 1200px/12000nm FOV), and
# (b) monotonically improving (not U-shaped) subpixel slope/residual as
# sigma rises across the whole tested range, with no sign of the
# non-monotonic "worse at high sigma" behavior seen in an earlier, less
# controlled sweep -- see docs/LOG.md and D-021. 8.0 nm is the best of the
# three tested values with no evidence of a downside; it is NOT chosen
# because it makes test 7 pass (it does not fully clear the existing
# thresholds -- see the actual rerun numbers in docs/LOG.md/results.json).
_SEARCH_PSF_SIGMA_NM = 8.0

# Production coarse/search (10 nm/px) supersample factor. A k-sweep (test 1's
# matched-PSF config, psf_sigma_nm=1.0) showed quadrature error converging
# monotonically: k=8 -> 2.178 DN, k=16 -> 0.791, k=32 -> 0.265, k=64 -> 0.097
# (threshold 2.0 DN) -- see docs/LOG.md and D-020/D-022. k=16 already clears
# the threshold with >2.5x margin at half the render cost of k=32, so it is
# the production value; k=32/k=64 are kept only as convergence-validation
# references (run on demand, not on every test invocation) confirming k=16
# has converged rather than landed on a lucky plateau.
_PRODUCTION_SEARCH_SUPERSAMPLE = 16


def _noiseless_capture_cfg(pixel_size_nm: float, size_px: int, psf_sigma_nm: float, supersample: int) -> CaptureConfig:
    return CaptureConfig(
        pixel_size_nm=pixel_size_nm,
        size_px=size_px,
        supersample=supersample,
        psf_sigma_nm=psf_sigma_nm,
        dose_e_per_px=_NOISELESS_DOSE,
        read_noise_sigma=0.0,
        brightness_offset=0.0,
        contrast_gain=1.0,
        row_band_amp=0.0,
        quantise_8bit=False,
    )


# ---------------------------------------------------------------------------
# Test 1 -- physical correspondence
# ---------------------------------------------------------------------------


def test_1_physical_correspondence():
    """Render the same 1 um x 1 um region noiseless at 1 nm/px and at
    10 nm/px (matched PSF/gain, per L0), area-average the 1 nm/px render
    10x10, and require mean absolute difference < 2 DN. This isolates the
    sampling/integration pipeline's self-consistency from any effect of
    differing optical configuration (matched PSF here, not each capture's
    normal production defaults).
    """
    geom_cfg = GeometryConfig(preset="intel14")
    model = build_geometry(geom_cfg, extent_nm=1500.0, rng=np.random.default_rng(0))

    fov_nm = 1000.0
    origin_nm = (-fov_nm / 2.0, -fov_nm / 2.0)
    psf_sigma_nm = 1.0

    # coarse_cfg uses _PRODUCTION_SEARCH_SUPERSAMPLE (16), not the
    # CaptureConfig/fine_cfg default of 8: at this test's matched
    # psf_sigma_nm=1.0 on a 10 nm pixel, k=8 gives
    # sigma_fine_px = 1.0/(10/8) = 0.8 -- the PSF's own sigma is *smaller
    # than one fine-pixel spacing*, a severely under-resolved discrete
    # approximation of the continuous Gaussian blur applied by
    # `gaussian_filter`. This is pure quadrature error, confirmed by a
    # k-sweep converging monotonically toward zero (k=8: 2.178 DN, k=16:
    # 0.791, k=32: 0.265, k=64: 0.097) -- see docs/LOG.md and D-020/D-022.
    # Not raised for fine_cfg: at 1 nm/px, sigma_fine_px = 1.0/(1/8) = 8
    # already, well resolved.
    fine_cfg = _noiseless_capture_cfg(1.0, 1000, psf_sigma_nm, supersample=8)
    coarse_cfg = _noiseless_capture_cfg(10.0, 100, psf_sigma_nm, supersample=_PRODUCTION_SEARCH_SUPERSAMPLE)

    fine_yield = render(model, origin_nm, fov_nm, fine_cfg.size_px, psf_sigma_nm, fine_cfg.supersample)
    coarse_yield = render(model, origin_nm, fov_nm, coarse_cfg.size_px, psf_sigma_nm, coarse_cfg.supersample)

    fine_dn = apply_detector(fine_yield, fine_cfg, np.random.default_rng(1))
    coarse_dn = apply_detector(coarse_yield, coarse_cfg, np.random.default_rng(2))

    fine_avg = fine_dn.reshape(100, 10, 100, 10).mean(axis=(1, 3))
    mad = float(np.mean(np.abs(fine_avg - coarse_dn)))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result("test1_physical_correspondence", {"mean_abs_diff_dn": mad, "threshold_dn": 2.0, "passed": mad < 2.0})

    assert mad < 2.0, f"mean abs diff {mad:.4f} DN >= 2.0 DN threshold"


# ---------------------------------------------------------------------------
# Test 6 -- subpixel shift recoverable
# ---------------------------------------------------------------------------


def _render_search_only(model, search_cfg: CaptureConfig, origin_nm: tuple[float, float]) -> np.ndarray:
    fov_nm = search_cfg.pixel_size_nm * search_cfg.size_px
    yield_img = render(
        model, origin_nm, fov_nm, search_cfg.size_px, search_cfg.psf_sigma_nm, search_cfg.supersample
    )
    return apply_detector(yield_img, search_cfg, np.random.default_rng(1))


def test_6_subpixel_shift_recoverable():
    """Two search-image renders of the same shared geometry, differing only
    by a rigid 0.2 search-px (= 2 nm) shift of where the search window
    looks (see pair.py D-019), must recover a 0.200 +/- 0.01 px shift via
    phase_cross_correlation(upsample_factor=100). This is the one
    permitted use of a correlation function in Phase 1 -- a measuring tool
    here, not a matcher.
    """
    geom_cfg = GeometryConfig(preset="intel14")
    search_cfg = _noiseless_capture_cfg(
        10.0, 200, psf_sigma_nm=_SEARCH_PSF_SIGMA_NM, supersample=_PRODUCTION_SEARCH_SUPERSAMPLE
    )
    fov_nm = search_cfg.pixel_size_nm * search_cfg.size_px

    model = build_geometry(geom_cfg, extent_nm=fov_nm / 2.0 + 100.0, rng=np.random.default_rng(0))
    origin_a = (-fov_nm / 2.0, -fov_nm / 2.0)

    true_shift_px = 0.2
    dx_nm = true_shift_px * search_cfg.pixel_size_nm
    origin_b = (origin_a[0] + dx_nm, origin_a[1])

    img_a = _render_search_only(model, search_cfg, origin_a)
    img_b = _render_search_only(model, search_cfg, origin_b)

    shift, _error, _diffphase = phase_cross_correlation(img_a, img_b, upsample_factor=100)
    recovered_x_px = float(abs(shift[1]))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test6_subpixel_shift",
        {"true_shift_px": true_shift_px, "recovered_shift_px": recovered_x_px},
    )

    assert abs(recovered_x_px - 0.200) <= 0.01, (
        f"recovered {recovered_x_px:.4f} px, expected 0.200 +/- 0.01"
    )


# ---------------------------------------------------------------------------
# Test 7 -- subpixel linearity
# ---------------------------------------------------------------------------


def test_7_subpixel_linearity():
    """Sweep the shift 0 -> 1.0 search px in 0.05 steps; fit recovered vs
    true, require slope 1.00 +/- 0.02 and no staircase (residual peak-to-
    peak well under the 0.05 px step size -- a staircase would produce
    residuals comparable to the step itself, since many sweep points would
    alias onto the same discrete recovered value).
    """
    geom_cfg = GeometryConfig(preset="intel14")
    search_cfg = _noiseless_capture_cfg(
        10.0, 200, psf_sigma_nm=_SEARCH_PSF_SIGMA_NM, supersample=_PRODUCTION_SEARCH_SUPERSAMPLE
    )
    fov_nm = search_cfg.pixel_size_nm * search_cfg.size_px

    model = build_geometry(geom_cfg, extent_nm=fov_nm / 2.0 + 100.0, rng=np.random.default_rng(0))
    origin0 = (-fov_nm / 2.0, -fov_nm / 2.0)
    img0 = _render_search_only(model, search_cfg, origin0)

    true_shifts = np.arange(0.0, 1.0 + 1e-9, 0.05)
    recovered = []
    for t in true_shifts:
        dx_nm = float(t) * search_cfg.pixel_size_nm
        origin_t = (origin0[0] + dx_nm, origin0[1])
        img_t = _render_search_only(model, search_cfg, origin_t)
        shift, _e, _d = phase_cross_correlation(img0, img_t, upsample_factor=100)
        recovered.append(float(abs(shift[1])))
    recovered = np.array(recovered)

    slope, intercept = np.polyfit(true_shifts, recovered, 1)
    residuals = recovered - (slope * true_shifts + intercept)
    residual_ptp = float(residuals.max() - residuals.min())

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test7_subpixel_linearity",
        {
            "true_shifts_px": true_shifts.tolist(),
            "recovered_px": recovered.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_ptp_px": residual_ptp,
        },
    )
    _plot_test7(true_shifts, recovered, slope, intercept)

    assert abs(slope - 1.0) <= 0.02, f"slope {slope:.4f}, expected 1.00 +/- 0.02"
    assert residual_ptp < 0.025, (
        f"residual peak-to-peak {residual_ptp:.4f} px looks staircased "
        f"(step size is 0.05 px) -- consider raising supersample"
    )


def _plot_test7(true_shifts, recovered, slope, intercept):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(true_shifts, recovered, "o-", ms=3)
    axes[0].plot(true_shifts, slope * true_shifts + intercept, "--", color="gray", lw=1, label=f"fit slope={slope:.3f}")
    axes[0].plot([0, 1], [0, 1], ":", color="red", lw=1, label="ideal slope=1")
    axes[0].set_xlabel("true shift (search px)")
    axes[0].set_ylabel("recovered shift (search px)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Test 7: recovered vs true subpixel shift")

    residuals = recovered - (slope * true_shifts + intercept)
    axes[1].plot(true_shifts, residuals, "o-", ms=3, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("true shift (search px)")
    axes[1].set_ylabel("residual (search px)")
    axes[1].set_title("Residuals (no staircase = smooth, small)")

    fig.tight_layout()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VALIDATION_DIR / "test7_subpixel_linearity.png", dpi=135)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 2 -- dose -> SNR
# ---------------------------------------------------------------------------


def test_2_dose_snr():
    """Sweep dose_e_per_px 50 -> 5000 on a flat (single-value) patch,
    Poisson noise only (no read noise/banding, so SNR reflects shot noise
    alone), and require SNR = mean/std to be proportional to sqrt(dose):
    SNR/sqrt(dose) must be constant within 5% across the sweep (D-015's
    own derivation: mean_electrons=yield*dose, Poisson std=sqrt(mean), so
    SNR = mean_dn/std_dn = sqrt(yield*dose), independent of dose only after
    dividing by sqrt(dose)).
    """
    doses = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])
    yield_flat_value = 0.817  # Y_SUBSTRATE == Y_FEATURE baseline
    n_samples = 20000  # keeps sampling noise in the estimated std small (~0.5%)

    snrs = []
    for i, dose in enumerate(doses):
        cfg = CaptureConfig(
            pixel_size_nm=10.0,
            size_px=1,
            supersample=1,
            psf_sigma_nm=1.0,
            dose_e_per_px=float(dose),
            read_noise_sigma=0.0,
            brightness_offset=0.0,
            contrast_gain=1.0,
            row_band_amp=0.0,
            quantise_8bit=False,
        )
        yield_patch = np.full(n_samples, yield_flat_value)
        dn = apply_detector(yield_patch, cfg, np.random.default_rng(100 + i))
        snrs.append(float(dn.mean() / dn.std()))
    snrs = np.array(snrs)

    ratio = snrs / np.sqrt(doses)
    rel_spread = float((ratio.max() - ratio.min()) / ratio.mean())

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test2_dose_snr",
        {"doses": doses.tolist(), "snr": snrs.tolist(), "snr_over_sqrt_dose": ratio.tolist(), "rel_spread": rel_spread},
    )

    assert rel_spread < 0.05, f"SNR/sqrt(dose) varies by {rel_spread:.4f} (>5%) across dose 50-5000"


# ---------------------------------------------------------------------------
# Shared isolated-single-edge helper (tests 3, 4)
# ---------------------------------------------------------------------------

# Isolated-edge fixture geometry (preset="custom", see D-025): the real
# intel14 fin is only 8 nm wide, so its own opposite edge sits just 8 nm
# from the edge under test -- for psf_sigma_nm up to CLAUDE.md's test-3
# ceiling of 4 nm (3 sigma = 12 nm), that opposite edge's own PSF response
# bleeds into the measurement window and breaks the "isolated single edge"
# premise (edge widths stopped increasing monotonically and even hit 0 at
# sigma=4, and test 4's peak/plateau ratio came out inflated). This is not
# a renderer bug -- an 8 nm fin genuinely cannot show a clean interior
# plateau once the beam PSF approaches half its width; that is a true
# physical effect. It just means the real intel14 fin is the wrong fixture
# for measuring an isolated edge response at these sigmas. fin_pitch/width
# are widened to 250/60 nm purely to give ~50 nm clearance to the nearest
# other edge (far beyond any swept sigma's influence); fin_height_nm,
# gate_pitch_nm, gate_length_nm are kept at intel14's values so the yield
# model's fin-height-dependent backscatter term (LAMBDA_B) stays
# physically representative. This fixture is used only by tests 3 and 4;
# no production preset uses preset="custom".
# _ISOLATED_EDGE_X_NM is for this custom fixture only (tests 3, 4) -- do
# not confuse with test 9's _EDGE_TEST_X_NM, which probes the real
# intel14 preset's fin edge at a different x.
_ISOLATED_EDGE_X_NM = 30.0  # fin idx=0 center=0, half-width=30 -> right edge at 30 nm
_EDGE_TEST_Y_NM = 35.0  # gate_pitch/2, keeps the gate term deep negative here


def _isolated_edge_model(ler_sigma3_nm: float = 0.0):
    geom_cfg = GeometryConfig(
        preset="custom",
        fin_pitch_nm=250.0,
        fin_width_nm=60.0,
        fin_height_nm=42.0,
        gate_pitch_nm=70.0,
        gate_length_nm=20.0,
        ler_sigma3_nm=ler_sigma3_nm,
        cut_density=0.0,
        landmark_count=0,
    )
    model = build_geometry(geom_cfg, extent_nm=200.0, rng=np.random.default_rng(0))
    return geom_cfg, model


# ---------------------------------------------------------------------------
# Test 3 -- PSF broadens edges
# ---------------------------------------------------------------------------


def test_3_psf_broadens_edges():
    """Isolated single fin edge (custom-preset fixture, LER disabled to
    isolate the PSF's own effect on edge shape from LER's independent
    y-direction wander), sweep psf_sigma_nm 0.5 -> 4.0 nm at reference
    scale (noiseless), measure the edge transition width; must increase
    strictly monotonically.

    Width metric: peak-to-trough x-separation, not a 20-80% rise between
    profile.min()/max(). Direct numeric inspection (see D-025) showed why
    the naive min/max version fails: this yield model has
    Y_SUBSTRATE == Y_FEATURE (0.817 both), so there is no bulk contrast --
    the entire signal is a transient bright peak just inside the edge
    (ALPHA_E/ALPHA_V) and a dark trough just outside it (PHI_F/PHI_B).
    profile.min()/max() pick up exactly that peak and trough, and PSF
    blur shrinks their *amplitude* toward the flat baseline even as it
    correctly grows their *separation* -- so a 20-80%-of-(max-min)
    crossing distance is not monotonic in sigma even though the edge is
    genuinely broadening. Direct measurement confirms peak-trough
    separation increases cleanly and monotonically (2.10, 3.70, 6.70,
    12.50 nm at sigma = 0.5, 1.0, 2.0, 4.0 nm) -- that is the metric this
    test uses.
    """
    _, model = _isolated_edge_model(ler_sigma3_nm=0.0)

    # Asymmetric window: 20 nm into the fin interior, 30 nm into the
    # substrate -- wide enough that neither the peak nor the trough sits
    # against a window boundary even at the largest swept sigma (4 nm).
    interior_margin_nm, substrate_margin_nm = 20.0, 30.0
    fov_nm = interior_margin_nm + substrate_margin_nm
    origin_nm = (_ISOLATED_EDGE_X_NM - interior_margin_nm, _EDGE_TEST_Y_NM - fov_nm / 2.0)
    out_px = int(fov_nm / 0.1)  # 0.1 nm/px

    # k=4 here (not the production k=16): at 0.1 nm/px output resolution
    # the fine step is already far smaller than any swept psf_sigma_nm
    # (>=0.5 nm), so quadrature error is negligible well below k=16 --
    # and gaussian_filter's cost scales with sigma in *fine* pixels
    # (psf_sigma_nm / fine_step_nm), which k=16 blows up to hundreds of
    # fine pixels here, making the sweep computationally infeasible for
    # no accuracy benefit. See D-025.
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    pixel_size_nm = fov_nm / out_px
    widths_nm = []
    for sigma in sigmas:
        yield_img = render(model, origin_nm, fov_nm, out_px, sigma, k=4)
        profile = yield_img[out_px // 2, :]
        idx_peak = int(np.argmax(profile))
        idx_trough = int(np.argmin(profile))
        widths_nm.append(abs(idx_trough - idx_peak) * pixel_size_nm)
    widths_nm = np.array(widths_nm)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test3_psf_broadens_edges",
        {"sigmas_nm": sigmas, "edge_widths_nm": widths_nm.tolist(), "monotonic": bool(np.all(np.diff(widths_nm) > 0))},
    )

    assert np.all(np.diff(widths_nm) > 0), f"edge widths not monotonically increasing: {widths_nm}"


# ---------------------------------------------------------------------------
# Test 4 -- edge response amplitude
# ---------------------------------------------------------------------------


def test_4_edge_response_amplitude():
    """Isolated single fin edge (same technique as test 3), toggle
    `edge_amp_scale` 0.0 -> 1.0. `edge_amp_scale` scales ALPHA_E together
    with ALPHA_V/PHI_F/PHI_B (D-013) rather than this test monkeypatching
    `yield_field.ALPHA_E` directly -- toggling the documented per-capture
    knob exercises the same "no edge enhancement" -> "nominal Mack-Bunday"
    transition CLAUDE.md describes without mutating a module-global
    constant that other tests/threads could observe. Measures the
    peak-to-plateau ratio on the rendered (PSF-blurred, pixel-integrated)
    profile -- not the raw pre-PSF analytic yield, whose true peak is a
    much sharper, higher spike that a real image never shows once the beam
    PSF and pixel integration are applied (D-012). Must rise from ~1.0 (no
    enhancement) to a substantially elevated ratio.
    """
    _, model = _isolated_edge_model(ler_sigma3_nm=0.0)

    # Same window as test 3 (D-025): margins wide enough that both ends
    # sit in the true asymptotic plateau, not on the transition's flank.
    interior_margin_nm, substrate_margin_nm = 20.0, 30.0
    fov_nm = interior_margin_nm + substrate_margin_nm
    origin_nm = (_ISOLATED_EDGE_X_NM - interior_margin_nm, _EDGE_TEST_Y_NM - fov_nm / 2.0)
    out_px = int(fov_nm / 0.1)
    psf_sigma_nm = 1.0

    ratios = {}
    for edge_amp_scale in (0.0, 1.0):
        # k=4, same rationale as test 3 (D-025): 0.1 nm/px output already
        # far finer than psf_sigma_nm=1.0, k=16 was needless and made
        # gaussian_filter's cost explode without changing the result.
        yield_img = render(model, origin_nm, fov_nm, out_px, psf_sigma_nm, k=4, edge_amp_scale=edge_amp_scale)
        profile = yield_img[out_px // 2, :]
        peak = float(profile.max())
        plateau = float(profile[-20:].mean())  # deep substrate side, far from the edge
        ratios[edge_amp_scale] = peak / plateau

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result("test4_edge_response_amplitude", {"ratio_at_0": ratios[0.0], "ratio_at_1": ratios[1.0]})

    assert abs(ratios[0.0] - 1.0) < 0.05, f"ratio at edge_amp_scale=0 should be ~1.0, got {ratios[0.0]:.4f}"
    assert ratios[1.0] > ratios[0.0], "peak-to-plateau ratio should rise with edge_amp_scale"
    assert 1.2 < ratios[1.0] < 2.5, f"ratio at edge_amp_scale=1.0 out of the expected elevated-but-blurred range: {ratios[1.0]:.4f}"


# ---------------------------------------------------------------------------
# Test 5 -- banding is row-structured
# ---------------------------------------------------------------------------


def test_5_banding_row_structured():
    """Flat (constant-yield) image with row_band_amp=4.0 (L5's value) plus
    a small amount of independent read noise (needed so column-means have
    *some* genuine variation to test for flatness against -- with zero
    other noise, a flat base image's column-means would be exactly
    constant, an uninformative degenerate case, not a real "is it flat"
    check). Row-mean autocorrelation must show the injected correlation
    length (`detector._BAND_CORR_LEN_PX`); column-mean autocorrelation
    must decay immediately (banding is row-broadcast, so averaging over
    many independent rows for a fixed column washes it out).
    """
    size = 300
    yield_flat = np.full((size, size), 0.817)
    cfg = CaptureConfig(
        pixel_size_nm=10.0,
        size_px=size,
        supersample=1,
        psf_sigma_nm=1.0,
        dose_e_per_px=_NOISELESS_DOSE,
        read_noise_sigma=1.0,
        brightness_offset=0.0,
        contrast_gain=1.0,
        row_band_amp=4.0,
        quantise_8bit=False,
    )
    dn = apply_detector(yield_flat, cfg, np.random.default_rng(42))

    def normalized_autocorr(x: np.ndarray) -> np.ndarray:
        x = x - x.mean()
        ac = np.correlate(x, x, mode="full")
        ac = ac[ac.size // 2 :]
        return ac / ac[0]

    row_ac = normalized_autocorr(dn.mean(axis=1))
    col_ac = normalized_autocorr(dn.mean(axis=0))

    row_corr_len_px = float(np.argmax(row_ac < 1.0 / np.e))
    col_lag1 = float(abs(col_ac[1]))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test5_banding_row_structured",
        {
            "row_autocorr_1e_lag_px": row_corr_len_px,
            "expected_corr_len_px": _BAND_CORR_LEN_PX,
            "col_autocorr_lag1": col_lag1,
        },
    )

    assert row_corr_len_px > 3.0, (
        f"row-mean autocorrelation decayed too fast (1/e lag={row_corr_len_px} px), banding not row-structured"
    )
    assert col_lag1 < 0.3, f"column-mean autocorrelation shows unexpected structure (lag1={col_lag1:.3f})"


# ---------------------------------------------------------------------------
# Test 8 -- no moire
# ---------------------------------------------------------------------------


def test_8_no_moire():
    """`cut_density=0` (perfectly periodic lattice): the FFT of a noiseless
    search-scale render must show power concentrated at the fin
    fundamental spatial frequency (and its harmonics), not at a spurious
    low frequency -- a low-frequency peak would indicate a moire/aliasing
    beat between the fin pitch and the pixel grid.
    """
    geom_cfg = GeometryConfig(preset="intel14", cut_density=0.0, landmark_count=0)
    pixel_size_nm = 10.0
    out_px = 400
    fov_nm = pixel_size_nm * out_px
    model = build_geometry(geom_cfg, extent_nm=fov_nm / 2.0 + 100.0, rng=np.random.default_rng(0))
    origin_nm = (-fov_nm / 2.0, -fov_nm / 2.0)

    cfg = CaptureConfig(
        pixel_size_nm=pixel_size_nm,
        size_px=out_px,
        supersample=8,
        psf_sigma_nm=_SEARCH_PSF_SIGMA_NM,
        dose_e_per_px=_NOISELESS_DOSE,
        read_noise_sigma=0.0,
        brightness_offset=0.0,
        contrast_gain=1.0,
        row_band_amp=0.0,
        quantise_8bit=False,
    )
    yield_img = render(model, origin_nm, fov_nm, out_px, cfg.psf_sigma_nm, cfg.supersample)
    dn = apply_detector(yield_img, cfg, np.random.default_rng(7))

    # Power spectrum along the fin-periodicity axis (x): average |FFT|^2
    # over rows.
    spectrum_rows = np.abs(np.fft.rfft(dn, axis=1)) ** 2
    mean_power = spectrum_rows.mean(axis=0)
    freqs = np.fft.rfftfreq(out_px, d=1.0)  # cycles/pixel

    fin_pitch_nm = 42.0  # intel14
    fundamental_cpp = pixel_size_nm / fin_pitch_nm  # cycles/pixel

    low_freq_mask = (freqs > 0.01) & (freqs < fundamental_cpp * 0.5)
    fundamental_mask = np.abs(freqs - fundamental_cpp) < 0.02

    low_freq_power = float(mean_power[low_freq_mask].max()) if low_freq_mask.any() else 0.0
    fundamental_power = float(mean_power[fundamental_mask].max()) if fundamental_mask.any() else 0.0

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test8_no_moire",
        {
            "fundamental_cycles_per_px": fundamental_cpp,
            "fundamental_power": fundamental_power,
            "low_freq_power": low_freq_power,
            "ratio": (low_freq_power / fundamental_power) if fundamental_power > 0 else None,
        },
    )

    assert fundamental_power > 0, "no detectable power at the fin fundamental frequency"
    assert low_freq_power < 0.1 * fundamental_power, (
        f"low-frequency power ({low_freq_power:.3e}) not negligible relative to the fin "
        f"fundamental ({fundamental_power:.3e}) -- possible moire/aliasing beat"
    )


# ---------------------------------------------------------------------------
# Test 9 -- shared LER
# ---------------------------------------------------------------------------


def _subpixel_edge_profile(image: np.ndarray, x_axis_nm: np.ndarray, x_lo_nm: float, x_hi_nm: float) -> np.ndarray:
    """Per-row sub-pixel edge x-position via a gradient-magnitude centroid
    within [x_lo_nm, x_hi_nm] -- a standard centroid-of-gradient edge
    localizer, chosen because a simple per-pixel argmax cannot resolve
    LER's ~3 nm amplitude at the search capture's 10 nm pixels (the
    centroid can, the same way phase_cross_correlation recovers sub-pixel
    shifts from smooth pixel-integrated data).
    """
    mask = (x_axis_nm >= x_lo_nm) & (x_axis_nm <= x_hi_nm)
    x_win = x_axis_nm[mask]
    img_win = image[:, mask].astype(np.float64)
    grad = np.abs(np.diff(img_win, axis=1))
    x_grad = 0.5 * (x_win[:-1] + x_win[1:])
    weight = grad.sum(axis=1)
    weight = np.where(weight > 0, weight, 1.0)
    return (grad * x_grad[np.newaxis, :]).sum(axis=1) / weight


# Real intel14 fin right edge (idx=1): center 42 nm, half-width 4 nm ->
# edge at 46 nm. Test-9 specific (unlike tests 3/4's widened custom
# fixture, D-025) -- test 9 needs the real preset's actual LER
# realisation, not an isolation fixture, so it uses intel14 directly.
_EDGE_TEST_X_NM = 46.0


def test_9_shared_ler():
    """Extract a sub-pixel edge-displacement profile for the same fin edge
    from each capture (one shared `GeometryModel`, per CLAUDE.md
    constraint #4); the two profiles (reference resampled onto search's
    coarser y-grid) must be positively cross-correlated. Zero/negative
    would mean the two captures somehow saw independently-resampled LER --
    a shared-geometry bug, since architecturally only one `GeometryModel`
    is ever built per pair (see `pair.py`).
    """
    geom_cfg = GeometryConfig(preset="intel14", cut_density=0.0, landmark_count=0)
    model = build_geometry(geom_cfg, extent_nm=400.0, rng=np.random.default_rng(0))

    fov_nm = 320.0
    origin_nm = (_EDGE_TEST_X_NM - fov_nm / 2.0, -fov_nm / 2.0)

    ref_pixel, ref_k = 1.0, 16
    search_pixel, search_k = 10.0, 8
    ref_out_px = int(fov_nm / ref_pixel)
    search_out_px = int(fov_nm / search_pixel)

    ref_yield = render(model, origin_nm, fov_nm, ref_out_px, 1.0, ref_k)
    search_yield = render(model, origin_nm, fov_nm, search_out_px, _SEARCH_PSF_SIGMA_NM, search_k)

    ref_x = origin_nm[0] + (np.arange(ref_out_px) + 0.5) * ref_pixel
    ref_y = origin_nm[1] + (np.arange(ref_out_px) + 0.5) * ref_pixel
    search_x = origin_nm[0] + (np.arange(search_out_px) + 0.5) * search_pixel
    search_y = origin_nm[1] + (np.arange(search_out_px) + 0.5) * search_pixel

    # Window: a few nm into the fin interior through well into the
    # substrate, wide enough for search's coarse pixels to contribute
    # multiple gradient samples, short of the next edge (8 nm to this
    # fin's own opposite edge, 42 nm to the next fin).
    x_lo, x_hi = _EDGE_TEST_X_NM - 3.0, _EDGE_TEST_X_NM + 30.0
    ref_edge = _subpixel_edge_profile(ref_yield, ref_x, x_lo, x_hi)
    search_edge = _subpixel_edge_profile(search_yield, search_x, x_lo, x_hi)

    ref_edge_on_search_grid = np.interp(search_y, ref_y, ref_edge)
    r = float(np.corrcoef(ref_edge_on_search_grid, search_edge)[0, 1])

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result("test9_shared_ler", {"cross_correlation": r})

    assert r > 0.15, (
        f"edge-profile cross-correlation between reference and search is {r:.4f}, "
        f"expected clearly positive (shared LER) -- ~0 would indicate independently-resampled geometry"
    )


# ---------------------------------------------------------------------------
# Test 10 -- determinism
# ---------------------------------------------------------------------------


def test_10_determinism():
    """Same `PairConfig` (including seed) run through `generate_pair`
    twice must produce bit-identical reference/search images and an
    identical truth dict. Uses small images (not full 1000 px) purely for
    test speed -- nothing about RNG stream-splitting depends on image
    size -- but keeps real noise/banding active (not the noiseless gate
    configs) since that is the part whose determinism is actually at
    stake.
    """
    geom_cfg = GeometryConfig(preset="intel14")
    reference = CaptureConfig(
        pixel_size_nm=1.0, size_px=50, supersample=4, psf_sigma_nm=1.0,
        dose_e_per_px=800.0, read_noise_sigma=2.0, quantise_8bit=True,
    )
    search = CaptureConfig(
        pixel_size_nm=10.0, size_px=50, supersample=4, psf_sigma_nm=8.0,
        dose_e_per_px=300.0, read_noise_sigma=2.0, row_band_amp=4.0, quantise_8bit=True,
    )
    cfg = PairConfig(geometry=geom_cfg, reference=reference, search=search, seed=12345)

    ref1, search1, truth1 = generate_pair(cfg)
    ref2, search2, truth2 = generate_pair(cfg)

    ref_identical = bool(np.array_equal(ref1, ref2))
    search_identical = bool(np.array_equal(search1, search2))
    truth_identical = truth1 == truth2

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result(
        "test10_determinism",
        {"ref_identical": ref_identical, "search_identical": search_identical, "truth_identical": truth_identical},
    )

    assert ref_identical, "reference image differs between two runs with the same seed"
    assert search_identical, "search image differs between two runs with the same seed"
    assert truth_identical, "truth dict differs between two runs with the same seed"


# ---------------------------------------------------------------------------
# Test 11 -- independence
# ---------------------------------------------------------------------------


def test_11_independence():
    """Same geometry (one shared render), two different noise seeds:
    correlation between the two noise *residuals* (each detected image
    minus a shared, effectively-noiseless expectation) must be small
    (|r| < 0.05). Correlating the raw detected images directly would be
    dominated by the shared signal (same edges/features in both) and
    would not test noise independence at all.
    """
    geom_cfg = GeometryConfig(preset="intel14")
    search_cfg_noisy = CaptureConfig(
        pixel_size_nm=10.0, size_px=200, supersample=8, psf_sigma_nm=8.0,
        dose_e_per_px=300.0, read_noise_sigma=2.0, quantise_8bit=False,
    )
    search_cfg_clean = replace(search_cfg_noisy, dose_e_per_px=_NOISELESS_DOSE, read_noise_sigma=0.0)

    fov_nm = search_cfg_noisy.pixel_size_nm * search_cfg_noisy.size_px
    origin_nm = (-fov_nm / 2.0, -fov_nm / 2.0)
    model = build_geometry(geom_cfg, extent_nm=fov_nm / 2.0 + 100.0, rng=np.random.default_rng(0))
    yield_img = render(
        model, origin_nm, fov_nm, search_cfg_noisy.size_px, search_cfg_noisy.psf_sigma_nm, search_cfg_noisy.supersample
    )

    clean_dn = apply_detector(yield_img, search_cfg_clean, np.random.default_rng(0))
    dn_a = apply_detector(yield_img, search_cfg_noisy, np.random.default_rng(101))
    dn_b = apply_detector(yield_img, search_cfg_noisy, np.random.default_rng(202))

    residual_a = (dn_a - clean_dn).ravel()
    residual_b = (dn_b - clean_dn).ravel()
    r = float(np.corrcoef(residual_a, residual_b)[0, 1])

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_result("test11_independence", {"noise_correlation": r})

    assert abs(r) < 0.05, (
        f"noise correlation |r|={abs(r):.4f} between independently-seeded captures of the "
        f"same geometry, expected < 0.05"
    )


# ---------------------------------------------------------------------------
# results.json bookkeeping
# ---------------------------------------------------------------------------


def _write_result(name: str, payload: dict) -> None:
    results_path = VALIDATION_DIR / "results.json"
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text())
    results[name] = payload
    results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
