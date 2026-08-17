"""Secondary-electron yield from signed distance (Mack-Bunday ALM,
two-exponential model).

Converts the geometry's signed distance field (nm, positive inside a
feature) into a dimensionless SE yield map. This is a piecewise-analytic
function of `d` alone, plus `fin_height_nm` (sets the backscatter range)
and `edge_amp_scale` (a per-capture knob, `CaptureConfig.edge_amp_scale`).
No rasterisation or PSF happens here -- this is step 2 of the forward model
order (geometry -> SE yield -> PSF -> pixel-area integration -> detector);
see optics.py for what comes next.
"""

from __future__ import annotations

import numpy as np

# ASSUMPTION: these are the Mack-Bunday analytical linescan model (ALM)
# constants for silicon-on-silicon at 500 eV landing energy, 90-degree
# sidewall, 20-100 nm feature height (tag (a) sourced, see docs/sources.md).
# Applying them to a generic FinFET stack (Si fin on STI oxide, HKMG gate --
# not silicon-on-silicon) is an approximation we have not calibrated against
# real tool data. Do not present this generator's contrast as a calibrated
# match to any specific tool or material stack.
Y_SUBSTRATE = 0.817
Y_FEATURE = 0.817
ALPHA_E = 1.65  # edge enhancement factor
SIGMA_E = 2.66  # nm, step forward-scatter range
ALPHA_V = 0.64  # volume loss factor
SIGMA_V = 0.26  # nm, volume loss range
PHI_F = 0.245  # substrate forward-scatter absorption
LAMBDA_F = 1.95  # nm, wafer forward-scatter range
PHI_B = 0.22  # substrate backscatter absorption
_LAMBDA_B_PER_HEIGHT_NM = 0.82  # backscatter range per nm of step height (unitless)


def se_yield(d_nm: np.ndarray, fin_height_nm: float, edge_amp_scale: float = 1.0) -> np.ndarray:
    """Secondary-electron yield at each point, from signed distance.

    `d_nm`: signed distance in nm (positive inside a feature), any NumPy
    array shape -- this is the direct output of `geometry.signed_distance`.

    `fin_height_nm`: sets the backscatter range
    `LAMBDA_B = 0.82 * fin_height_nm` (nm); the model has no separate
    gate-height parameter, per CLAUDE.md.

    `edge_amp_scale`: independent per-capture multiplier (see
    `CaptureConfig.edge_amp_scale`) on the *edge-related* excess over the
    `Y_FEATURE`/`Y_SUBSTRATE` baseline -- it scales `ALPHA_E`, `ALPHA_V`,
    `PHI_F`, and `PHI_B` together, leaving the flat far-field baseline
    plateaus unaffected. `1.0` = nominal Mack-Bunday values, `0.0` collapses
    the field to a flat `Y_SUBSTRATE == Y_FEATURE` plane.

    Returns the dimensionless SE yield, same shape as `d_nm`. Note: at
    exactly `d = 0` this piecewise formula is discontinuous (jumps from the
    outside branch to the inside branch) -- see D-012, this is an expected
    property of the model, not a bug, and gets smoothed by the beam PSF in
    the next forward-model stage.
    """
    lambda_b_nm = _LAMBDA_B_PER_HEIGHT_NM * fin_height_nm

    # np.where always evaluates both branches over the full array before
    # selecting; SIGMA_V = 0.26 nm is fast enough that exp(-d_nm/SIGMA_V)
    # overflows for d_nm below roughly -184 nm, in the branch that gets
    # discarded for those (outside) points anyway. Harmless (np.where's
    # selection drops it cleanly, never propagates into the output -- see
    # step 5 validation), but silence the resulting warning explicitly
    # rather than let real problems hide in the noise later.
    with np.errstate(over="ignore", invalid="ignore"):
        y_inside = (
            Y_FEATURE
            + edge_amp_scale * ALPHA_E * np.exp(-d_nm / SIGMA_E)
            - edge_amp_scale * ALPHA_V * np.exp(-d_nm / SIGMA_V)
        )
        y_outside = (
            Y_SUBSTRATE
            - edge_amp_scale * PHI_F * np.exp(d_nm / LAMBDA_F)
            - edge_amp_scale * PHI_B * np.exp(d_nm / lambda_b_nm)
        )
    return np.where(d_nm > 0, y_inside, y_outside)
