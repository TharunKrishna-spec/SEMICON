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

Current scope (build step 3): sharp corners, no cuts, no landmarks yet --
those are added in later build steps.
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
    n_features = 2 * int(np.ceil(extent_nm / pitch_nm)) + 3
    index_min = -(n_features // 2)
    n_samples = 2 * int(np.ceil(extent_nm / _LER_SAMPLE_STEP_NM)) + 3
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
class GeometryModel:
    """An immutable, realised FinFET lattice in physical nm, including its
    LER realisation.

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


def build_geometry(
    cfg: GeometryConfig, extent_nm: float, rng: np.random.Generator
) -> GeometryModel:
    """Resolve `cfg.preset` and realise a GeometryModel from it, including
    one LER draw, valid for queries within [-extent_nm, extent_nm]^2.

    `cfg` may carry an unresolved `preset` name (fin/gate dimensions filled
    from NODE_PRESETS); the returned model always has concrete nm values.

    `rng` is the geometry-realisation RNG stream -- callers must spawn it
    separately from any per-capture noise stream (CLAUDE.md constraint #4:
    geometry/LER are shared between captures, noise is independent).
    """
    resolved = resolve_preset(cfg)
    fin_ler_left, fin_ler_right = _make_edge_roughness_pair(
        resolved, extent_nm, resolved.fin_pitch_nm, rng
    )
    gate_ler_bottom, gate_ler_top = _make_edge_roughness_pair(
        resolved, extent_nm, resolved.gate_pitch_nm, rng
    )
    return GeometryModel(
        cfg=resolved,
        rotation_rad=np.deg2rad(resolved.rotation_deg),
        extent_nm=extent_nm,
        fin_ler_left=fin_ler_left,
        fin_ler_right=fin_ler_right,
        gate_ler_bottom=gate_ler_bottom,
        gate_ler_top=gate_ler_top,
    )


def signed_distance(model: GeometryModel, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Signed distance in nm from each query point to the nearest fin/gate
    edge, positive inside a feature, negative in the substrate.

    `X`, `Y` are nm coordinates in the die's world frame, any broadcastable
    NumPy array shapes (this is the vectorised, per-pixel-loop-free query
    the optics module supersamples over).
    """
    cfg = model.cfg
    theta = model.rotation_rad

    # Rotate the query points into the lattice's local (unrotated) frame --
    # rotating the query point, not the rectangles, is what keeps this
    # correct for |rotation_deg| > 0: the fin/gate stripes below are defined
    # as axis-aligned in this local frame regardless of rotation_deg.
    c, s = np.cos(theta), np.sin(theta)
    lx = X * c + Y * s
    ly = -X * s + Y * c

    fin_half_w = cfg.fin_width_nm / 2.0
    fin_idx = np.round(lx / cfg.fin_pitch_nm)
    fin_center_x = fin_idx * cfg.fin_pitch_nm
    left_out = model.fin_ler_left.sample(fin_idx, ly)
    right_out = model.fin_ler_right.sample(fin_idx, ly)
    dist_left = (lx - fin_center_x + fin_half_w) + left_out
    dist_right = (fin_center_x + fin_half_w - lx) + right_out
    d_fin = np.minimum(dist_left, dist_right)

    gate_half_w = cfg.gate_length_nm / 2.0
    gate_idx = np.round(ly / cfg.gate_pitch_nm)
    gate_center_y = gate_idx * cfg.gate_pitch_nm
    bottom_out = model.gate_ler_bottom.sample(gate_idx, lx)
    top_out = model.gate_ler_top.sample(gate_idx, lx)
    dist_bottom = (ly - gate_center_y + gate_half_w) + bottom_out
    dist_top = (gate_center_y + gate_half_w - ly) + top_out
    d_gate = np.minimum(dist_bottom, dist_top)

    # Union of fins and gates: in the positive-inside convention, union is
    # the max of the two signed distances (inside either => positive).
    # ASSUMPTION: exact only away from fin/gate crossings; near a crossing
    # corner this slightly overestimates distance versus a true 2-D rounded
    # union. This is the standard SDF-union approximation and only matters
    # within ~one edge-response width of a corner.
    return np.maximum(d_fin, d_gate)
