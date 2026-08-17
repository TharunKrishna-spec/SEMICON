"""L0-L6 difficulty ladder + cut_density sweep (CLAUDE.md Step 12).

Each level function returns a fully-specified `PairConfig` (minus `seed`,
which callers assign per case) built from a shared reference/search
baseline, with each level adding exactly the deltas CLAUDE.md's ladder
table specifies on top of the previous level -- L0 is the easiest
(noiseless, matched gain), L6 the hardest (all asymmetries active, no
landmarks, perfectly periodic geometry, reference placed at the lattice
origin).
"""

from __future__ import annotations

from dataclasses import replace

from .config import CaptureConfig, GeometryConfig, PairConfig

# Production defaults established during Phase 1 gate validation (see
# docs/DECISIONS.md D-020/D-021/D-022/D-024), used as the L0-L3 baseline
# here:
# - search psf_sigma_nm = 8.0 nm, not CLAUDE.md's literal 3.0 nm default --
#   a boundary-controlled sweep found 8.0 nm the best-performing,
#   no-boundary-artifact option among {3,5,8} nm for the search capture's
#   subpixel-recovery behavior at 10 nm/px. Tests 6/7 still show a
#   documented residual subpixel bias at this value (D-021) -- accepted as
#   a known Phase 1 limitation, not fixed by a threshold or code change.
# - supersample = 8, NOT the validation gate's k=16 (D-020/D-022). k=16 is
#   numerically better (confirmed on the gate's small 100-200 px images)
#   but is not memory-feasible at CLAUDE.md's required full 1000x1000 px
#   output: the fine grid is out_px*k per side, so 1000 px at k=16 is
#   16000x16000 -- a single such float64 array is 1.9 GiB, and
#   `signed_distance()` holds many simultaneously (~10-15 GiB+ peak),
#   which OOMs in this environment (~10 GB available). k=8 (8000x8000 fine
#   grid) renders successfully (~103 s reference + ~82 s search per pair)
#   and is independently confirmed numerically adequate for *these*
#   specific psf/pixel combinations: sigma_fine_px = psf_sigma_nm * k /
#   pixel_size_nm = 1.0*8/1.0 = 8 (reference) and 8.0*8/10.0 = 6.4
#   (search), both comfortably above the ~1 threshold where D-020 found
#   quadrature error become significant (that under-resolution was
#   specific to test 1's contrived matched-psf_sigma_nm=1.0-at-10nm/px
#   configuration, which no L0-L6 preset uses). See D-024.
_REF_PSF_SIGMA_NM = 1.0
_SEARCH_PSF_SIGMA_NM = 8.0
_PRODUCTION_SUPERSAMPLE = 8
_NOISELESS_DOSE = 1.0e12  # "dose -> infinity": Poisson noise negligible, not skipped

# ASSUMPTION: CLAUDE.md's L4 rung widens search psf_sigma_nm from its
# (then-)3.0 nm default to 3.5 nm -- a +0.5 nm ladder-escalation step. Since
# the search baseline used here is 8.0 nm (D-021), not 3.0 nm, applying the
# literal "3.5" at L4 would make L4 *narrower* (less blurred, and per our
# own sigma sweep, *more* subpixel-aliased) than the L0-L3 baseline --
# inverting the ladder's intended monotonic-difficulty structure. Instead,
# L4-L6 preserve CLAUDE.md's original *relative* step (+0.5 nm) applied to
# the revised baseline: 8.0 + 0.5 = 8.5 nm. See D-023.
_SEARCH_PSF_SIGMA_NM_L4PLUS = _SEARCH_PSF_SIGMA_NM + 0.5

CUT_DENSITY_SWEEP = (0.05, 0.02, 0.01, 0.005, 0.0)


def _geometry(**overrides) -> GeometryConfig:
    return GeometryConfig(preset="intel14", **overrides)


def _reference(**overrides) -> CaptureConfig:
    cfg = CaptureConfig(
        pixel_size_nm=1.0,
        size_px=1000,
        supersample=_PRODUCTION_SUPERSAMPLE,
        psf_sigma_nm=_REF_PSF_SIGMA_NM,
        dose_e_per_px=_NOISELESS_DOSE,
        read_noise_sigma=0.0,
        brightness_offset=0.0,
        contrast_gain=1.0,
        edge_amp_scale=1.0,
        row_band_amp=0.0,
        quantise_8bit=True,
    )
    return replace(cfg, **overrides)


def _search(**overrides) -> CaptureConfig:
    cfg = CaptureConfig(
        pixel_size_nm=10.0,
        size_px=1000,
        supersample=_PRODUCTION_SUPERSAMPLE,
        psf_sigma_nm=_SEARCH_PSF_SIGMA_NM,
        dose_e_per_px=_NOISELESS_DOSE,
        read_noise_sigma=0.0,
        brightness_offset=0.0,
        contrast_gain=1.0,
        edge_amp_scale=1.0,
        row_band_amp=0.0,
        quantise_8bit=True,
    )
    return replace(cfg, **overrides)


def L0(seed: int = 0) -> PairConfig:
    """Noiseless, matched gain, PSF on, dose -> infinity (Poisson skipped)."""
    return PairConfig(geometry=_geometry(), reference=_reference(), search=_search(), seed=seed)


def L1(seed: int = 0) -> PairConfig:
    """L0 + Poisson/Gaussian noise, symmetric dose=800 both captures."""
    return PairConfig(
        geometry=_geometry(),
        reference=_reference(dose_e_per_px=800.0, read_noise_sigma=2.0),
        search=_search(dose_e_per_px=800.0, read_noise_sigma=2.0),
        seed=seed,
    )


def L2(seed: int = 0) -> PairConfig:
    """L1 + asymmetric dose: ref 2000, search 300."""
    return PairConfig(
        geometry=_geometry(),
        reference=_reference(dose_e_per_px=2000.0, read_noise_sigma=2.0),
        search=_search(dose_e_per_px=300.0, read_noise_sigma=2.0),
        seed=seed,
    )


def L3(seed: int = 0) -> PairConfig:
    """L2 + search brightness_offset=15, contrast_gain=0.75."""
    return PairConfig(
        geometry=_geometry(),
        reference=_reference(dose_e_per_px=2000.0, read_noise_sigma=2.0),
        search=_search(
            dose_e_per_px=300.0, read_noise_sigma=2.0, brightness_offset=15.0, contrast_gain=0.75
        ),
        seed=seed,
    )


def L4(seed: int = 0) -> PairConfig:
    """L3 + ref/search PSF divergence and edge_amp_scale asymmetry
    (0.9 reference / 1.1 search -- an arbitrary assignment, see D-023;
    Phase 2 is not expected to know which capture got which multiplier).
    """
    return PairConfig(
        geometry=_geometry(),
        reference=_reference(
            dose_e_per_px=2000.0,
            read_noise_sigma=2.0,
            psf_sigma_nm=_REF_PSF_SIGMA_NM,
            edge_amp_scale=0.9,
        ),
        search=_search(
            dose_e_per_px=300.0,
            read_noise_sigma=2.0,
            brightness_offset=15.0,
            contrast_gain=0.75,
            psf_sigma_nm=_SEARCH_PSF_SIGMA_NM_L4PLUS,
            edge_amp_scale=1.1,
        ),
        seed=seed,
    )


def L5(seed: int = 0) -> PairConfig:
    """L4 + search row_band_amp=4.0."""
    cfg = L4(seed=seed)
    return replace(cfg, search=replace(cfg.search, row_band_amp=4.0))


def L6(seed: int = 0) -> PairConfig:
    """L5 + cut_density=0 (perfectly periodic, intrinsically ambiguous),
    landmark_count=0 (no coarse anchor), reference forced to the lattice
    origin (0, 0) -- the hardest case in the ladder.
    """
    cfg = L5(seed=seed)
    geometry = replace(cfg.geometry, cut_density=0.0, landmark_count=0)
    return replace(cfg, geometry=geometry, ref_center_nm=(0.0, 0.0))


def cut_density_case(cut_density: float, seed: int = 0) -> PairConfig:
    """Cases at otherwise-fixed L2 settings, varying only
    `geometry.cut_density` -- for the Phase 3 failure-boundary
    characterisation (CLAUDE.md: sweep {0.05, 0.02, 0.01, 0.005, 0.0}).
    """
    cfg = L2(seed=seed)
    return replace(cfg, geometry=replace(cfg.geometry, cut_density=cut_density))


LEVEL_PRESETS = {
    "L0": L0,
    "L1": L1,
    "L2": L2,
    "L3": L3,
    "L4": L4,
    "L5": L5,
    "L6": L6,
}
