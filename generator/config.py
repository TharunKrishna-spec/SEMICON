"""Configuration dataclasses for the SEM pair generator.

Units: all `_nm` fields are physical nanometres, all `_px` fields are pixels,
all bare-DN fields (dose, offsets, noise sigmas, gains) are detector digital
numbers unless stated otherwise. See CLAUDE.md for the forward model order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ASSUMPTION: node presets below are taken from published/disclosed figures
# for the named process nodes but are not independently re-derived here.
# See docs/sources.md for provenance tags.
NODE_PRESETS: dict[str, dict[str, float]] = {
    "intel14": {
        "fin_pitch_nm": 42.0,
        "fin_width_nm": 8.0,
        "fin_height_nm": 42.0,
        "gate_pitch_nm": 70.0,
        "gate_length_nm": 20.0,
    },
    "n7": {
        "fin_pitch_nm": 30.0,
        "fin_width_nm": 6.0,
        "fin_height_nm": 45.0,
        "gate_pitch_nm": 48.0,
        "gate_length_nm": 18.0,
    },
    "n5": {
        "fin_pitch_nm": 28.0,
        "fin_width_nm": 6.0,
        "fin_height_nm": 50.0,
        "gate_pitch_nm": 44.0,
        "gate_length_nm": 16.0,
    },
}


@dataclass
class GeometryConfig:
    preset: str = "intel14"
    fin_pitch_nm: float = 42.0
    fin_width_nm: float = 8.0
    fin_height_nm: float = 42.0
    gate_pitch_nm: float = 70.0
    gate_length_nm: float = 20.0
    rotation_deg: float = 0.0  # keep small, 0-2
    corner_radius_nm: float = 2.0
    ler_sigma3_nm: float = 3.0
    ler_corr_len_nm: float = 25.0
    cut_density: float = 0.02  # 0 => intrinsically ambiguous
    landmark_count: int = 2
    landmark_scale_nm: float = 150.0


@dataclass
class CaptureConfig:
    pixel_size_nm: float  # nm per pixel; 1.0 for reference, 10.0 for search
    size_px: int = 1000
    supersample: int = 8
    psf_sigma_nm: float = 1.0
    dose_e_per_px: float = 800.0
    read_noise_sigma: float = 2.0  # DN
    brightness_offset: float = 0.0  # DN
    contrast_gain: float = 1.0
    edge_amp_scale: float = 1.0
    row_band_amp: float = 0.0  # DN, 0 disables
    quantise_8bit: bool = True


@dataclass
class PairConfig:
    geometry: GeometryConfig
    reference: CaptureConfig
    search: CaptureConfig
    ref_center_nm: tuple[float, float] | None = None  # None => random placement
    subpixel_offset_px: tuple[float, float] = (0.0, 0.0)
    seed: int = 0


def resolve_preset(geometry: GeometryConfig) -> GeometryConfig:
    """Return a copy of `geometry` with fin/gate pitch, width, height and
    length overwritten from `NODE_PRESETS[geometry.preset]`. All other
    fields (rotation, LER, cuts, landmarks) pass through unchanged.

    `preset="custom"` bypasses the NODE_PRESETS lookup entirely and returns
    `geometry` unchanged -- an escape hatch for callers (currently only
    validate.py's isolated-edge fixture, see D-025) that need fin/gate
    dimensions outside the three named process nodes. Not used by any
    production preset in presets.py.
    """
    if geometry.preset == "custom":
        return geometry
    if geometry.preset not in NODE_PRESETS:
        raise ValueError(
            f"unknown geometry preset {geometry.preset!r}; "
            f"expected one of {sorted(NODE_PRESETS)}"
        )
    values = NODE_PRESETS[geometry.preset]
    return replace(geometry, **values)
