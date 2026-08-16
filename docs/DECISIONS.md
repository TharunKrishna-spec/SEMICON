# Decision Record

Append-only. Never edit or delete a past entry; if a decision is reversed, add a new entry that supersedes it and say so.

## D-001 — Fin/gate lattice as index-based infinite stripes, not an explicit rectangle list
**Date:** 2026-08-16
**Status:** accepted
**Context:** `signed_distance(model, X, Y)` must be vectorised over NumPy arrays of
arbitrary shape (full 1000x1000 x supersample^2 grids), queried many times per
render. A FinFET die region can contain hundreds of fins and dozens of gates
within a 10 um search FOV.
**Decision:** Represent fins and gates as infinite periodic stripes (fins:
width `fin_width_nm`, pitch `fin_pitch_nm`, running along local y; gates:
width `gate_length_nm`, pitch `gate_pitch_nm`, running along local x). At each
query point, find the nearest stripe center via `round(coord / pitch) * pitch`
(O(1), independent of stripe count) rather than storing/broadcasting a finite
list of rectangle objects.
**Alternatives considered:** Explicit array of N fin rectangles and M gate
rectangles, computing signed distance to each and reducing (min/max) over the
array (O(N+M) per query point).
**Rationale:** The index-based approach is exact for a uniform periodic
lattice (no LER, no cuts yet) and is O(1) per query point instead of O(N+M),
which matters at supersampled render resolution. It also handles the search
image's 10 um FOV without pre-deciding how large a rectangle list to
generate.
**Consequences:** Assumes on-pitch periodicity. Once LER (step 3) and cuts
(step 4) are added, the index computed here (`round(coord/pitch)`) becomes
the natural "which fin/gate" key for looking up per-feature perturbations
(LER realisation, cut presence) — the design anticipates that reuse.
**Supersedes:** —

## D-002 — SDF union via max(), not an exact 2-D boolean union
**Date:** 2026-08-16
**Status:** accepted
**Context:** The feature is the union of the fin stripe and the gate stripe
(both raised, i.e. "inside" in the positive-inside convention). An exact
Euclidean SDF of a general union of two shapes requires care near where their
boundaries are close together.
**Decision:** Use `d_union = max(d_fin, d_gate)`, the standard approximate
SDF composition for a union in a positive-inside convention.
**Alternatives considered:** An exact rectilinear union SDF — possible here
since fins and gates are both axis-aligned in the local frame — but adds real
complexity for a discrepancy that only shows up within one edge-response
width of a fin/gate corner.
**Rationale:** `max()` is exact everywhere except asymptotically close to the
concave corners formed at fin/gate crossings, where it can overestimate
distance to the true nearest boundary point. This is the standard, widely
used approximation in SDF modelling and is adequate because the yield model
(Mack-Bunday) only responds meaningfully within a few `sigma_e`/`sigma_v`
(~1-3 nm) of an edge — the corner-region error is second-order relative to
that.
**Consequences:** Fin/gate crossing corners are not exactly rounded in signed
distance; this is a known, bounded approximation, not a bug. Revisit with an
exact rectilinear union if Phase 2 sensitivity analysis shows corner-region
error matters.
**Supersedes:** —

## D-003 — Lattice rotation applied once to query points about the origin, not per-rectangle
**Date:** 2026-08-16
**Status:** accepted
**Context:** `GeometryConfig.rotation_deg` represents a small (0-2 deg)
die/stage misalignment relative to the scan raster. It must rotate the whole
periodic lattice rigidly.
**Decision:** In `signed_distance`, rotate the query point (X, Y) by
`-rotation_deg` about the origin once, then evaluate fin/gate stripes as
axis-aligned in that local frame. There is no per-rectangle rotation
parameter.
**Alternatives considered:** Rotating each fin/gate rectangle individually
about its own center.
**Rationale:** Rotating individual rectangles about their own centers would
make them fan out relative to each other instead of staying parallel, which
does not match what a rotated real layout looks like. Rotating the query
point once is also the only formulation compatible with the index-based
(round-to-nearest-pitch) stripe lookup in D-001.
**Consequences:** None known.
**Supersedes:** —

## D-004 — Corner rounding deferred; fins/gates are sharp-cornered for now
**Date:** 2026-08-16
**Status:** accepted (temporary, by design)
**Context:** `GeometryConfig.corner_radius_nm` exists (default 2.0 nm) but
CLAUDE.md's build order for step 2 explicitly scopes `geometry.py` to "no
LER, no cuts" and does not call for corner rounding either.
**Decision:** Ship step 2 with sharp-cornered rectangles. `corner_radius_nm`
is currently unused.
**Alternatives considered:** Implementing rounded corners now via a
shrink-then-offset SDF trick, since it is a small amount of extra code.
**Rationale:** Corner rounding on the union-of-stripes formulation (D-002) is
not simply "round each rectangle" — the crossing geometry is not a single
rectangle, so getting it right needs its own small design pass and its own
eyeball check, which is out of scope for a "no LER, no cuts" step. Deferring
keeps step 2 reviewable on its own.
**Consequences:** `docs/validation/step2_sdf.png` shows sharp corners; do not
read that as final geometry.
**Supersedes:** —

## D-005 — `build_geometry()` resolves the preset internally
**Date:** 2026-08-16
**Status:** accepted
**Context:** `GeometryConfig.preset` needs to be expanded to concrete
fin/gate nm values (via `config.resolve_preset`) before geometry can be
built.
**Decision:** `build_geometry(cfg)` calls `resolve_preset(cfg)` internally
and stores the resolved config on `GeometryModel`; callers never need to
remember to resolve the preset themselves.
**Alternatives considered:** Requiring callers (e.g. `pair.py`) to call
`resolve_preset()` before `build_geometry()`.
**Rationale:** Centralising this removes a class of bug where a caller
passes an unresolved `GeometryConfig` (e.g. `preset="n7"` but still carrying
`intel14` default nm fields) straight into rendering.
**Consequences:** `GeometryModel.cfg` is always resolved; there is no way to
construct one with an unresolved preset.
**Supersedes:** —

## D-006 — Gaussian-correlated LER via smoothed white noise; correlation length = filter kernel std
**Date:** 2026-08-16
**Status:** accepted
**Context:** LER needs a spatial correlation structure. CD-SEM roughness
literature characterises LER via a power-law PSD split into low/mid/high
frequency bands (see CLAUDE.md's own worked example).
**Decision:** Sample white noise on a uniform grid (`_LER_SAMPLE_STEP_NM =
2.0`) per feature edge, run `scipy.ndimage.gaussian_filter1d` along the
running coordinate with `sigma = ler_corr_len_nm / _LER_SAMPLE_STEP_NM`
samples, then rescale so the realised array's empirical std equals
`ler_sigma3_nm / 3`. `ler_corr_len_nm` is treated directly as the Gaussian
filter kernel's standard deviation, not fit to a target autocorrelation
width (which for a Gaussian kernel convolved with itself would be
`sqrt(2) * sigma_kernel`).
**Alternatives considered:** Full power-law PSD (own validation burden, ~40
lines); white noise per edge point with no smoothing (would be washed out
entirely by the PSF, per CLAUDE.md's own framing).
**Rationale:** Power-law PSD is out of scope for the value it would add;
Gaussian correlation captures the property Phase 2 actually depends on (each
fin period is distinguishable at 1 nm/px but not at 10 nm/px). Treating
`ler_corr_len_nm` as the kernel std directly (rather than deriving it from a
target ACF width) is a simplification -- rescaling to the empirical std after
filtering guarantees the realised roughness amplitude is exactly right
regardless.
**Consequences:** The LER spectrum is not spectrally calibrated to a specific
ACF definition; `ler_corr_len_nm` is "a correlation length" in the loose,
qualitative sense, not a precisely fit one. If Phase 2 results turn out
sensitive to the exact roughness spectrum, revisit.
**Supersedes:** —

## D-007 — Fin/gate edges perturbed independently per side, not width-preserving
**Date:** 2026-08-16
**Status:** accepted
**Context:** A fin (or gate) has two edges. Real LER/LWR literature
distinguishes single-edge roughness (LER) from line-width roughness (LWR),
which depends on the correlation between the two edges of the same line.
**Decision:** Generate four independent `_EdgeRoughness` realisations
(fin-left, fin-right, gate-bottom, gate-top) with no correlation between the
two edges of the same feature. Local width can therefore vary as edges
wander independently.
**Alternatives considered:** Perturbing both edges of a feature identically
(width-preserving, only centerline wanders); modelling the left/right
correlation explicitly (adds a free parameter with no literature value
given in this project's scope).
**Rationale:** Independent per-edge roughness is the simpler, more standard
default assumption and is sufficient for what Phase 2 needs: edges that are
locally distinguishable at 1 nm/px and washed out at 10 nm/px. Modelling
edge-to-edge correlation would need its own literature-sourced correlation
coefficient, which CLAUDE.md does not provide.
**Consequences:** Local fin/gate width has more variance than a real
width-preserving process would. Not expected to matter for localisation
(which cares about edge position, not width per se); revisit only if Phase 2
results turn out sensitive to width variance specifically.
**Supersedes:** —

## D-008 — `build_geometry()` now requires a finite `extent_nm` and a geometry RNG stream
**Date:** 2026-08-16
**Status:** accepted
**Context:** LER (D-006) must be "generated once per model, in physical
coordinates, stored on the model object" (CLAUDE.md step 3), which requires
sampling white noise on a bounded grid -- unlike the index-based infinite
tiling used for the bare lattice (D-001), LER cannot be evaluated at an
unbounded coordinate without first committing to a finite realised domain.
**Decision:** `build_geometry(cfg, extent_nm, rng)` now takes the half-width
of a square domain `[-extent_nm, extent_nm]^2` (in the rotated local frame)
over which LER is realised, plus an `np.random.Generator` dedicated to
geometry realisation. `GeometryModel.extent_nm` records this so future code
(cuts/landmarks in step 4, and callers in `pair.py`) has one place to read
the valid domain from. Queries outside the domain will index out of bounds
and raise, by design (CLAUDE.md working style: don't validate what can't
happen -- it is `pair.py`'s job, not `signed_distance`'s, to size
`extent_nm` correctly for whatever field of view will actually be rendered).
**Alternatives considered:** A closed-form/procedural per-index noise
synthesis (e.g. summed sinusoids seeded by a hash of feature index) that
would work at unbounded coordinates without a finite domain.
**Rationale:** A closed-form synthesis that produces a true Gaussian-shaped
autocorrelation over an unbounded domain is materially more complex than
"filter some white noise," and CLAUDE.md's own dependency list
(`scipy.ndimage.gaussian_filter`/`gaussian_filter1d`) signals the smoothed-
white-noise approach is the intended one.
**Consequences:** `pair.py` (step 8) must size `extent_nm` to comfortably
cover the search capture's 10 um field of view (plus the reference capture,
which is nested inside it) before calling `build_geometry`; this is a real
constraint on that module's design, not yet implemented.
**Supersedes:** —
