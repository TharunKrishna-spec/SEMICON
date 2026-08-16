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

Current scope (build step 2): sharp-cornered rectangles only. No LER, no
cuts, no landmarks, no corner rounding yet -- those are added in later
build steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig, resolve_preset


@dataclass(frozen=True)
class GeometryModel:
    """An immutable, realised FinFET lattice in physical nm.

    Frozen and built once per die region so both captures in a pair can
    share exactly this object -- there is no way to hand the two captures
    different geometry (CLAUDE.md constraint #4).
    """

    cfg: GeometryConfig  # resolved: preset already expanded to concrete nm values
    rotation_rad: float


def build_geometry(cfg: GeometryConfig) -> GeometryModel:
    """Resolve `cfg.preset` and realise a GeometryModel from it.

    `cfg` may carry an unresolved `preset` name (fin/gate dimensions filled
    from NODE_PRESETS); the returned model always has concrete nm values.
    """
    resolved = resolve_preset(cfg)
    return GeometryModel(cfg=resolved, rotation_rad=np.deg2rad(resolved.rotation_deg))


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
    fin_center_x = np.round(lx / cfg.fin_pitch_nm) * cfg.fin_pitch_nm
    d_fin = fin_half_w - np.abs(lx - fin_center_x)

    gate_half_w = cfg.gate_length_nm / 2.0
    gate_center_y = np.round(ly / cfg.gate_pitch_nm) * cfg.gate_pitch_nm
    d_gate = gate_half_w - np.abs(ly - gate_center_y)

    # Union of fins and gates: in the positive-inside convention, union is
    # the max of the two signed distances (inside either => positive).
    # ASSUMPTION: exact only away from fin/gate crossings; near a crossing
    # corner this slightly overestimates distance versus a true 2-D rounded
    # union. This is the standard SDF-union approximation and only matters
    # within ~one edge-response width of a corner.
    return np.maximum(d_fin, d_gate)
