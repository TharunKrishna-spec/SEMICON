"""Analytic FinFET geometry and signed-distance queries.

All coordinates and lengths here are physical nanometres. `signed_distance`
is the only query the rest of the pipeline is allowed to make into this
module: distance in nm to the nearest feature edge at an arbitrary
continuous (x, y), positive inside a feature (fin or gate), negative in the
substrate. No rasterisation happens in this module.

Layout convention: fins are stripes of width `fin_width_nm` running along
the local y-axis, spaced `fin_pitch_nm` apart along the local x-axis. Gates
are stripes of width `gate_length_nm` running along the local x-axis, spaced
`gate_pitch_nm` apart along the local y-axis, crossing over the fins. The
whole lattice is rotated rigidly about the origin by `rotation_deg` (kept
small, 0-2 degrees) to represent die/stage misalignment relative to the scan
raster -- individual fins and gates are never rotated independently, or the
lattice would fan out instead of staying parallel.

Line-edge roughness (LER) perturbs each of the four edge families (fin-left,
fin-right, gate-bottom, gate-top) independently as a function of position
along the edge, realised once per GeometryModel over a finite domain and
shared by both captures of a pair (CLAUDE.md constraint #4).

Fin/gate cuts remove a short segment of a feature at candidate sites in the
gaps between the perpendicular feature's crossings (see D-009), breaking the
lattice's translational symmetry -- with cut_density=0 the pattern is
perfectly periodic and localisation within one period is intrinsically
ambiguous, which is the point of this knob.

Landmarks are large sharp-edged square pads at random positions, meant as
strong, easily-matched anchors distinct from the fine fin/gate texture.

Current scope (build step 4): sharp corners only -- corner rounding
(`corner_radius_nm`) is still deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import GeometryConfig, resolve_preset

# ASSUMPTION: the LER sampling grid step is fixed at 2 nm, chosen to resolve
# the default 25 nm correlation length with ~12 samples/length. If
# `ler_corr_len_nm` is set well below ~20 nm this under-resolves the
# correlation and the realised roughness will look coarser/blockier than a
# true continuum Gaussian-correlated process.
_LER_SAMPLE_STEP_NM = 2.0

# ASSUMPTION: a cut's length along the feature it severs is set to 30% of
# the *perpendicular* feature's pitch (gate_pitch for a fin cut, fin_pitch
# for a gate cut) -- i.e. relative to the gap it has to fit inside, not to
# its own feature's width. This is engineering judgement (no literature
# value given); see D-010.
_CUT_LENGTH_FRACTION = 0.3

# ASSUMPTION: landmarks are placed uniformly at random within 80% of
# extent_nm (a margin so a landmark's own footprint doesn't hang off the
# edge of the realised domain); see D-011.
_LANDMARK_PLACEMENT_MARGIN = 0.8


def _index_range(extent_nm: float, pitch_nm: float) -> tuple[int, int]:
    """Integer index range [index_min, index_min + n) covering
    round(coord / pitch - phase) for any coord in [-extent_nm, extent_nm]
    and any phase in [0, 1) (a small margin is included so both
    feature-center indices, phase=0, and cut-site indices, phase=0.5, fit).
    """
    n = 2 * int(np.ceil(extent_nm / pitch_nm)) + 3
    index_min = -(n // 2)
    return index_min, n


def _box_sdf(qx: np.ndarray, qy: np.ndarray, half_w: float, half_h: float) -> np.ndarray:
    """Signed distance, positive inside, from (qx, qy) to an axis-aligned
    box of half-width `half_w` and half-height `half_h` centered at the
    origin of the (qx, qy) frame (caller offsets the query point first).
    """
    dx = np.abs(qx) - half_w
    dy = np.abs(qy) - half_h
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return -(outside + inside)


@dataclass(frozen=True)
class _EdgeRoughness:
    """Precomputed Gaussian-correlated per-edge perturbation, in nm, sampled
    on a uniform grid along one edge family's running coordinate, for every
    feature (fin or gate) index in range.

    Positive values bulge the edge outward (widening the feature); this is
    the offset added to the nominal (unperturbed) edge position before
    computing distance -- see `signed_distance`.
    """

    values: np.ndarray  # (n_features, n_samples), nm
    index_min: int  # feature index (round(coord/pitch)) of row 0
    coord0_nm: float  # running-coordinate value of column 0
    coord_step_nm: float  # spacing between columns, nm

    def sample(self, feature_idx: np.ndarray, coord_nm: np.ndarray) -> np.ndarray:
        """Linearly interpolate the perturbation at each query point.

        `feature_idx` (integer-valued float or int array) and `coord_nm`
        must be the same (broadcast) shape.
        """
        row = feature_idx.astype(np.int64) - self.index_min
        col_f = (coord_nm - self.coord0_nm) / self.coord_step_nm
        col0 = np.clip(np.floor(col_f).astype(np.int64), 0, self.values.shape[1] - 2)
        frac = col_f - col0
        v0 = self.values[row, col0]
        v1 = self.values[row, col0 + 1]
        return v0 * (1.0 - frac) + v1 * frac


def _make_edge_roughness_pair(
    cfg: GeometryConfig, extent_nm: float, pitch_nm: float, rng: np.random.Generator
) -> tuple[_EdgeRoughness, _EdgeRoughness]:
    """Build the two independent _EdgeRoughness realisations (e.g. left and
    right for a fin family) for one feature family spaced at `pitch_nm`,
    valid over the square domain [-extent_nm, extent_nm]^2.
    """
    sigma_nm = cfg.ler_sigma3_nm / 3.0
    index_min, n_features = _index_range(extent_nm, pitch_nm)
    _, n_samples = _index_range(extent_nm, _LER_SAMPLE_STEP_NM)
    coord0_nm = -(n_samples // 2) * _LER_SAMPLE_STEP_NM
    sigma_samples = cfg.ler_corr_len_nm / _LER_SAMPLE_STEP_NM

    def make_one() -> _EdgeRoughness:
        raw = rng.standard_normal((n_features, n_samples))
        smoothed = gaussian_filter1d(raw, sigma=sigma_samples, axis=1, mode="reflect")
        smoothed *= sigma_nm / smoothed.std()
        return _EdgeRoughness(
            values=smoothed,
            index_min=index_min,
            coord0_nm=coord0_nm,
            coord_step_nm=_LER_SAMPLE_STEP_NM,
        )

    return make_one(), make_one()


@dataclass(frozen=True)
class _CutSites:
    """Precomputed cut presence, one boolean per (feature index, candidate
    site index) pair. Candidate sites for a feature family sit at the
    midpoints between consecutive crossings of the *perpendicular* feature
    family (e.g. fin-cut sites sit halfway between consecutive gate rows) --
    see D-009. `site_pitch_nm` is that perpendicular feature's pitch.
    """

    present: np.ndarray  # (n_features, n_sites), bool
    feature_index_min: int
    site_index_min: int
    site_pitch_nm: float
    half_len_nm: float

    def notch_term(self, feature_idx: np.ndarray, running_coord_nm: np.ndarray) -> np.ndarray:
        """The `-notch_sdf` term to combine via `minimum(d_feature, ...)`
        (CSG subtraction: `min(d_A, -d_B)` removes B from A). Returns +inf
        wherever no cut is present at the nearest candidate site, so it has
        no effect there.
        """
        site_idx = np.round(running_coord_nm / self.site_pitch_nm - 0.5)
        site_center_nm = (site_idx + 0.5) * self.site_pitch_nm
        notch_sdf = self.half_len_nm - np.abs(running_coord_nm - site_center_nm)
        row = feature_idx.astype(np.int64) - self.feature_index_min
        col = site_idx.astype(np.int64) - self.site_index_min
        has_cut = self.present[row, col]
        return np.where(has_cut, -notch_sdf, np.inf)


def _make_cut_sites(
    cfg: GeometryConfig,
    extent_nm: float,
    feature_pitch_nm: float,
    site_pitch_nm: float,
    rng: np.random.Generator,
) -> _CutSites:
    """Build cut-site presence for one feature family (fins or gates),
    spaced at `feature_pitch_nm`, with candidate cut sites spaced at
    `site_pitch_nm` (the perpendicular family's pitch).
    """
    feature_index_min, n_features = _index_range(extent_nm, feature_pitch_nm)
    site_index_min, n_sites = _index_range(extent_nm, site_pitch_nm)
    present = rng.random((n_features, n_sites)) < cfg.cut_density
    half_len_nm = 0.5 * _CUT_LENGTH_FRACTION * site_pitch_nm
    return _CutSites(
        present=present,
        feature_index_min=feature_index_min,
        site_index_min=site_index_min,
        site_pitch_nm=site_pitch_nm,
        half_len_nm=half_len_nm,
    )


def _make_landmarks(cfg: GeometryConfig, extent_nm: float, rng: np.random.Generator) -> np.ndarray:
    """Random landmark centers (n, 2) in the local frame, or shape (0, 2)
    if `landmark_count` is 0.
    """
    if cfg.landmark_count <= 0:
        return np.zeros((0, 2))
    lo, hi = -_LANDMARK_PLACEMENT_MARGIN * extent_nm, _LANDMARK_PLACEMENT_MARGIN * extent_nm
    return rng.uniform(lo, hi, size=(cfg.landmark_count, 2))


@dataclass(frozen=True)
class GeometryModel:
    """An immutable, realised FinFET lattice in physical nm, including its
    LER, cut, and landmark realisation.

    Frozen and built once per die region so both captures in a pair can
    share exactly this object -- there is no way to hand the two captures
    different geometry (CLAUDE.md constraint #4). Valid for signed_distance
    queries within [-extent_nm, extent_nm] in both x and y (in the rotated
    local frame); the caller that builds this model is responsible for
    sizing extent_nm to cover whatever field of view will be queried.
    """

    cfg: GeometryConfig  # resolved: preset already expanded to concrete nm values
    rotation_rad: float
    extent_nm: float
    fin_ler_left: _EdgeRoughness
    fin_ler_right: _EdgeRoughness
    gate_ler_bottom: _EdgeRoughness
    gate_ler_top: _EdgeRoughness
    fin_cuts: _CutSites
    gate_cuts: _CutSites
    landmark_centers_nm: np.ndarray  # (landmark_count, 2), local frame


def build_geometry(
    cfg: GeometryConfig, extent_nm: float, rng: np.random.Generator
) -> GeometryModel:
    """Resolve `cfg.preset` and realise a GeometryModel from it -- LER,
    cuts, and landmarks -- valid for queries within [-extent_nm, extent_nm]^2.

    `cfg` may carry an unresolved `preset` name (fin/gate dimensions filled
    from NODE_PRESETS); the returned model always has concrete nm values.

    `rng` is the geometry-realisation RNG stream -- callers must spawn it
    separately from any per-capture noise stream (CLAUDE.md constraint #4:
    geometry/LER/cuts/landmarks are shared between captures, noise is
    independent).
    """
    resolved = resolve_preset(cfg)
    fin_ler_left, fin_ler_right = _make_edge_roughness_pair(
        resolved, extent_nm, resolved.fin_pitch_nm, rng
    )
    gate_ler_bottom, gate_ler_top = _make_edge_roughness_pair(
        resolved, extent_nm, resolved.gate_pitch_nm, rng
    )
    fin_cuts = _make_cut_sites(
        resolved, extent_nm, resolved.fin_pitch_nm, resolved.gate_pitch_nm, rng
    )
    gate_cuts = _make_cut_sites(
        resolved, extent_nm, resolved.gate_pitch_nm, resolved.fin_pitch_nm, rng
    )
    landmark_centers_nm = _make_landmarks(resolved, extent_nm, rng)
    return GeometryModel(
        cfg=resolved,
        rotation_rad=np.deg2rad(resolved.rotation_deg),
        extent_nm=extent_nm,
        fin_ler_left=fin_ler_left,
        fin_ler_right=fin_ler_right,
        gate_ler_bottom=gate_ler_bottom,
        gate_ler_top=gate_ler_top,
        fin_cuts=fin_cuts,
        gate_cuts=gate_cuts,
        landmark_centers_nm=landmark_centers_nm,
    )


def to_local_frame(
    model: GeometryModel, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world-frame nm coordinates (X, Y) into the lattice's local
    (unrotated) frame -- the frame fins/gates are axis-aligned in, and the
    frame `landmark_centers_nm` is expressed in. Public because callers
    outside this module (e.g. `pair.py`'s lattice-phase/landmark-distance
    diagnostics) need the same transform `signed_distance` uses internally.
    """
    theta = model.rotation_rad
    c, s = np.cos(theta), np.sin(theta)
    return X * c + Y * s, -X * s + Y * c


def signed_distance(model: GeometryModel, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Signed distance in nm from each query point to the nearest fin/gate/
    landmark edge, positive inside a feature, negative in the substrate.

    `X`, `Y` are nm coordinates in the die's world frame, any broadcastable
    NumPy array shapes (this is the vectorised, per-pixel-loop-free query
    the optics module supersamples over).
    """
    cfg = model.cfg
    lx, ly = to_local_frame(model, X, Y)

    fin_half_w = cfg.fin_width_nm / 2.0
    fin_idx = np.round(lx / cfg.fin_pitch_nm)
    fin_center_x = fin_idx * cfg.fin_pitch_nm
    left_out = model.fin_ler_left.sample(fin_idx, ly)
    right_out = model.fin_ler_right.sample(fin_idx, ly)
    dist_left = (lx - fin_center_x + fin_half_w) + left_out
    dist_right = (fin_center_x + fin_half_w - lx) + right_out
    d_fin = np.minimum(dist_left, dist_right)
    d_fin = np.minimum(d_fin, model.fin_cuts.notch_term(fin_idx, ly))

    gate_half_w = cfg.gate_length_nm / 2.0
    gate_idx = np.round(ly / cfg.gate_pitch_nm)
    gate_center_y = gate_idx * cfg.gate_pitch_nm
    bottom_out = model.gate_ler_bottom.sample(gate_idx, lx)
    top_out = model.gate_ler_top.sample(gate_idx, lx)
    dist_bottom = (ly - gate_center_y + gate_half_w) + bottom_out
    dist_top = (gate_center_y + gate_half_w - ly) + top_out
    d_gate = np.minimum(dist_bottom, dist_top)
    d_gate = np.minimum(d_gate, model.gate_cuts.notch_term(gate_idx, lx))

    # Union of fins and gates: in the positive-inside convention, union is
    # the max of the two signed distances (inside either => positive).
    # ASSUMPTION: exact only away from fin/gate crossings and cut/landmark
    # boundaries; near such a corner this slightly overestimates distance
    # versus a true 2-D rounded union/subtraction. This is the standard
    # SDF-composition approximation and only matters within ~one
    # edge-response width of a corner.
    d = np.maximum(d_fin, d_gate)

    half_scale = cfg.landmark_scale_nm / 2.0
    for cx, cy in model.landmark_centers_nm:
        d = np.maximum(d, _box_sdf(lx - cx, ly - cy, half_scale, half_scale))

    return d
