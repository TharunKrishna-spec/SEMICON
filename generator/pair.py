"""Pairs: one shared geometry realisation, two independently rendered and
detected captures, and the ground truth linking them.

`generate_pair(cfg)` is the single entry point tying `geometry.py`,
`yield_field.py` (via `optics.render`), `optics.py`, and `detector.py`
together, and is where CLAUDE.md constraint #4 (shared geometry, independent
everything-else) is enforced structurally: exactly one `GeometryModel` is
built and passed to both `optics.render` calls, and reference/search each
get their own spawned RNG stream for detector noise -- there is no code
path by which the two captures could see different geometry.
"""

from __future__ import annotations

import numpy as np

from .config import PairConfig
from .detector import apply_detector
from .geometry import build_geometry, to_local_frame
from .optics import render

# ASSUMPTION: the search capture's field of view is anchored (centered) at
# world (0, 0) rather than also being randomly placed -- reference-center
# placement (and geometry's own LER/cut/landmark realisation) already vary
# per case, so this is a simplifying, defensible default rather than a
# requirement. See D-018.
_EXTENT_MARGIN_NM = 100.0
_REF_PLACEMENT_MARGIN_NM = 50.0


def generate_pair(cfg: PairConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one (reference, search) capture pair and its ground truth.

    Returns `(ref_image, search_image, truth)`:
    - `ref_image`: `(reference.size_px, reference.size_px)` array, `uint8`
      if `cfg.reference.quantise_8bit` else `float64` (see D-017).
    - `search_image`: same, at `cfg.search.size_px` / `cfg.search`'s dtype.
    - `truth`: dict with `ref_center_nm`, `search_origin_nm` (both nm,
      world frame), `gt_search_px` (float, search-pixel units -- where the
      reference center falls within the search image), and diagnostics
      `lattice_phase_fin`, `lattice_phase_gate` (fraction of one pitch,
      in `[-0.5, 0.5)`, signed offset from the nearest fin/gate centerline)
      and `dist_to_nearest_landmark_nm` (float, or `None` if
      `landmark_count == 0`).
    """
    base_rng = np.random.default_rng(cfg.seed)
    geom_rng, placement_rng, ref_noise_rng, search_noise_rng = base_rng.spawn(4)

    ref_fov_nm = cfg.reference.pixel_size_nm * cfg.reference.size_px
    search_fov_nm = cfg.search.pixel_size_nm * cfg.search.size_px

    extent_nm = search_fov_nm / 2.0 + _EXTENT_MARGIN_NM
    model = build_geometry(cfg.geometry, extent_nm=extent_nm, rng=geom_rng)

    # Search FOV is anchored centered at world (0, 0) -- see ASSUMPTION above.
    search_origin_nm = (-search_fov_nm / 2.0, -search_fov_nm / 2.0)

    if cfg.ref_center_nm is not None:
        ref_center_nm = cfg.ref_center_nm
        lo = -search_fov_nm / 2.0
        hi = search_fov_nm / 2.0
        half_ref = ref_fov_nm / 2.0
        if not (
            lo + half_ref <= ref_center_nm[0] <= hi - half_ref
            and lo + half_ref <= ref_center_nm[1] <= hi - half_ref
        ):
            raise ValueError(
                f"ref_center_nm={ref_center_nm} does not keep the "
                f"{ref_fov_nm:.0f} nm reference FOV fully inside the "
                f"{search_fov_nm:.0f} nm search FOV"
            )
    else:
        margin = ref_fov_nm / 2.0 + _REF_PLACEMENT_MARGIN_NM
        lo = -search_fov_nm / 2.0 + margin
        hi = search_fov_nm / 2.0 - margin
        rx, ry = placement_rng.uniform(lo, hi, size=2)
        ref_center_nm = (float(rx), float(ry))

    # subpixel_offset_px: a rigid shift (in search-pixel units) applied to
    # *both* ref_center_nm and search_origin_nm together -- i.e. it moves
    # where in the (shared) geometry we're looking, without changing the
    # reference's position relative to the search frame. This is what makes
    # a controlled sub-search-pixel content shift between two otherwise
    # identical renders (used by validation tests 6/7) -- see D-019.
    dx_nm = cfg.subpixel_offset_px[0] * cfg.search.pixel_size_nm
    dy_nm = cfg.subpixel_offset_px[1] * cfg.search.pixel_size_nm
    ref_center_nm = (ref_center_nm[0] + dx_nm, ref_center_nm[1] + dy_nm)
    search_origin_nm = (search_origin_nm[0] + dx_nm, search_origin_nm[1] + dy_nm)

    ref_origin_nm = (
        ref_center_nm[0] - ref_fov_nm / 2.0,
        ref_center_nm[1] - ref_fov_nm / 2.0,
    )

    ref_yield = render(
        model,
        ref_origin_nm,
        ref_fov_nm,
        cfg.reference.size_px,
        cfg.reference.psf_sigma_nm,
        cfg.reference.supersample,
        edge_amp_scale=cfg.reference.edge_amp_scale,
    )
    search_yield = render(
        model,
        search_origin_nm,
        search_fov_nm,
        cfg.search.size_px,
        cfg.search.psf_sigma_nm,
        cfg.search.supersample,
        edge_amp_scale=cfg.search.edge_amp_scale,
    )

    ref_image = apply_detector(ref_yield, cfg.reference, ref_noise_rng)
    search_image = apply_detector(search_yield, cfg.search, search_noise_rng)

    gt_search_px = (
        (ref_center_nm[0] - search_origin_nm[0]) / cfg.search.pixel_size_nm,
        (ref_center_nm[1] - search_origin_nm[1]) / cfg.search.pixel_size_nm,
    )

    lx, ly = to_local_frame(model, ref_center_nm[0], ref_center_nm[1])
    fin_pitch_nm = model.cfg.fin_pitch_nm
    gate_pitch_nm = model.cfg.gate_pitch_nm
    lattice_phase_fin = ((lx / fin_pitch_nm + 0.5) % 1.0) - 0.5
    lattice_phase_gate = ((ly / gate_pitch_nm + 0.5) % 1.0) - 0.5

    if model.landmark_centers_nm.shape[0] > 0:
        dists = np.hypot(
            model.landmark_centers_nm[:, 0] - lx, model.landmark_centers_nm[:, 1] - ly
        )
        dist_to_nearest_landmark_nm = float(dists.min())
    else:
        dist_to_nearest_landmark_nm = None

    truth = {
        "ref_center_nm": [ref_center_nm[0], ref_center_nm[1]],
        "search_origin_nm": [search_origin_nm[0], search_origin_nm[1]],
        "gt_search_px": [gt_search_px[0], gt_search_px[1]],
        "lattice_phase_fin": float(lattice_phase_fin),
        "lattice_phase_gate": float(lattice_phase_gate),
        "dist_to_nearest_landmark_nm": dist_to_nearest_landmark_nm,
    }

    return ref_image, search_image, truth
