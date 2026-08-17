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

## D-009 — Cut candidate sites at gap midpoints between perpendicular features, independent per-site Bernoulli
**Date:** 2026-08-16
**Status:** accepted
**Context:** `cut_density` needs to place breaks in the otherwise perfectly
periodic fin/gate lattice. CLAUDE.md motivates this directly: `cut_density=0`
is called out as "intrinsically ambiguous," i.e. cuts are what break
translational symmetry and make a region of the lattice locally unique.
**Decision:** For each feature (a specific fin, or a specific gate),
candidate cut sites sit at the midpoints between consecutive crossings of
the *perpendicular* feature family -- e.g. a fin's candidate sites are at
`y = (j + 0.5) * gate_pitch_nm` for integer `j` (halfway between gate rows).
Each (feature index, site index) pair independently has a cut with
probability `cut_density`, drawn once at `build_geometry` time from the
shared geometry RNG.
**Alternatives considered:** Cuts at arbitrary continuous positions (no
lattice alignment); a single global cut density applied as a Poisson process
in continuous space rather than a per-site Bernoulli.
**Rationale:** Real fin-cut / gate-cut ("CMG") litho steps sever a line
between two existing transistors, which is exactly the region between
perpendicular-feature crossings -- so anchoring candidate sites there is
physically motivated, not arbitrary. A per-site Bernoulli with the gap
lattice as the candidate set gives `cut_density` a direct, intuitive
reading: the fraction of gaps that are cut.
**Consequences:** Cuts always sit centered in a gap; they cannot occur at
arbitrary offsets within a gap. This is a modelling simplification (c) --
revisit if Phase 2 needs cut-position variability specifically.
**Supersedes:** —

## D-010 — Cut length is 30% of the perpendicular feature's pitch
**Date:** 2026-08-16
**Status:** accepted
**Context:** CLAUDE.md gives no explicit cut-length parameter or value.
**Decision:** A cut's full length (along the feature it severs) is
`0.3 * <perpendicular pitch>` -- e.g. a fin cut is `0.3 * gate_pitch_nm`
long. `_CUT_LENGTH_FRACTION = 0.3` in `geometry.py`.
**Alternatives considered:** Scaling cut length to the severed feature's own
width instead of the gap it sits in (tried first informally: for gate cuts
this produced notches close to or exceeding the fin pitch, physically
implausible -- swapped to scaling against the gap-defining pitch instead).
**Rationale:** 30% of the gap leaves comfortable clearance on both sides
(the notch does not reach the neighbouring crossings) while still being
unambiguously visible relative to typical edge-response widths (~1-3 nm).
**Consequences:** Purely an engineering choice (c), not sourced. If Phase 2
sensitivity analysis shows cut length matters, this is the knob to revisit.
**Supersedes:** —

## D-011 — Landmarks are sharp-edged axis-aligned square pads, no LER, placed uniformly at random
**Date:** 2026-08-16
**Status:** accepted
**Context:** `landmark_count` / `landmark_scale_nm` need a concrete shape and
placement rule; CLAUDE.md does not specify either, only that L6 disables
landmarks entirely to characterise the hardest failure case (no coarse
anchor, `cut_density=0`, reference forced near lattice center).
**Decision:** Each landmark is an axis-aligned (local frame) square of side
`landmark_scale_nm`, exact box SDF (`_box_sdf`), unioned into the feature set
via the same `max()` composition as fins/gates. Centers drawn uniformly at
random within 80% of `extent_nm` from the shared geometry RNG. No LER on
landmark edges.
**Alternatives considered:** A more distinctive asymmetric shape (L or cross)
to also aid orientation disambiguation; applying LER to landmark edges too.
**Rationale:** L6's framing ("no coarse anchor") implies landmarks' only
required role is being a strong, unique, easily-matched large-scale feature
-- a plain square is the simplest shape that does that. LER on a
150 nm-scale pad would be visually negligible (roughness sigma ~1 nm) and
not worth the extra code path.
**Consequences:** Landmarks carry no orientation information beyond
position; if Phase 2 wants rotation/orientation validation from landmark
shape specifically, this would need revisiting. Landmark placement doesn't
check for overlap with other landmarks or alignment to the fin/gate grid --
acceptable given the default count is small (0-3 across the L0-L6 ladder).
**Supersedes:** —

## D-012 — SE yield implemented exactly as specified; the d=0 discontinuity is kept, not smoothed
**Date:** 2026-08-16
**Status:** accepted
**Context:** CLAUDE.md gives the Mack-Bunday two-exponential yield formula
as two separate piecewise branches (`d > 0` vs `d <= 0`), each with its own
baseline and its own pair of exponential terms. Evaluating both branches at
`d = 0` gives materially different values (outside: `0.352`; inside limit as
`d -> 0+`: `1.827`) -- a genuine jump discontinuity, not a rounding
artefact.
**Decision:** Implement the formula exactly as given, with no blending or
continuity correction at `d = 0`.
**Alternatives considered:** Blending the two branches near `d=0` (e.g. a
narrow linear or cosine ramp) to force continuity.
**Rationale:** This is not a bug to fix -- it is the expected shape of a
real edge-yield response: SE escape geometry genuinely differs on the two
sides of a physical step edge, and real CD-SEM edge bloom comes from exactly
this kind of sharp near-edge response, softened into the smooth bright-edge
signal seen in real images only by the finite beam PSF (the next stage in
the forward model, step 6). Manufacturing a fake continuity fix here would
mean inventing physics not in the spec, and CLAUDE.md's working style is
explicit: implement what's specified, report what's observed, don't paper
over an unexpected result to make it "look right." Verified numerically:
far-field plateaus converge to `Y_SUBSTRATE == Y_FEATURE == 0.817` (slowly
on the substrate side -- `LAMBDA_B` scales with fin height, ~34 nm for
`intel14`, so full convergence needs several hundred nm); the outside branch
dips to `0.352` approaching an edge; the inside branch jumps to `1.827` and
overshoots to a true peak (~2.10 at `d~0.4 nm` for `intel14`) before
decaying back to baseline -- consistent with the "volume loss dip right at
the edge, edge-enhancement peak just inside" structure real ALM literature
describes.
**Consequences:** Pre-PSF yield maps have a hard bright/dark ring around
every feature edge (visible in `docs/validation/step5_yield.png`, panel 3).
This is intentional; do not "fix" it in `optics.py` -- the PSF convolution
is supposed to be what smooths it.
**Supersedes:** —

## D-013 — `edge_amp_scale` scales all four edge-excursion terms together, not just ALPHA_E
**Date:** 2026-08-16
**Status:** accepted
**Context:** `CaptureConfig.edge_amp_scale` is an independent per-capture
knob (L4: ref/search differ 0.9/1.1). CLAUDE.md doesn't specify exactly
which term(s) of the yield formula it should scale.
**Decision:** `edge_amp_scale` multiplies `ALPHA_E`, `ALPHA_V`, `PHI_F`, and
`PHI_B` together (i.e. the entire excess-over-baseline on both sides of an
edge), leaving `Y_SUBSTRATE`/`Y_FEATURE` (the flat far-field baseline)
untouched.
**Alternatives considered:** Scaling only `ALPHA_E` (the literal "edge
enhancement factor"), leaving `ALPHA_V`/`PHI_F`/`PHI_B` fixed.
**Rationale:** The config field is named for "edge amplitude" generally, not
"edge-enhancement-factor amplitude" specifically, and CLAUDE.md's own
validation test 4 (later, step 11) toggles the model constant `ALPHA_E`
directly to isolate its effect -- implying `edge_amp_scale` is meant to be a
distinct, coarser per-capture knob ("how much edge contrast does this
acquisition show overall") rather than a proxy for the same constant. An
asymmetric choice (scaling enhancement but not volume-loss, or forward-
scatter but not backscatter) would have no physical motivation given in
CLAUDE.md, so scaling everything edge-related uniformly is the simpler,
more defensible default.
**Consequences:** `edge_amp_scale=0` collapses the model to a flat
`Y_SUBSTRATE`/`Y_FEATURE` plane with no edge contrast at all (verified
numerically). If Phase 2 needs to isolate forward-scatter vs. backscatter vs.
volume-loss asymmetry specifically, this knob does not support that; would
need new, separate config fields.
**Supersedes:** —

## D-014 — `render()` takes an extra `edge_amp_scale` parameter beyond CLAUDE.md's literal signature; no FOV padding for PSF boundary handling
**Date:** 2026-08-16
**Status:** accepted
**Context:** CLAUDE.md's step-6 recipe literally lists
`render(model, origin_nm, fov_nm, out_px, psf_sigma_nm, k)`, but the SE
yield stage inside `render()` needs `edge_amp_scale`
(`CaptureConfig.edge_amp_scale`, independent per capture) to call
`yield_field.se_yield`. Separately, `gaussian_filter` needs a boundary
handling choice at the edge of the supersampled FOV.
**Decision:** Added `edge_amp_scale: float = 1.0` as an extra keyword
parameter (not in the literal listed signature). For the PSF boundary, used
scipy's default `mode="reflect"` directly on the nominal FOV, with **no**
extra margin queried/cropped around it.
**Alternatives considered:** Threading a whole `CaptureConfig` into
`render()` instead of separate scalars (rejected: pulls detector-only
fields -- dose, read noise, banding -- into the optics module, blurring the
`optics.py`/`detector.py` split). Querying a padded region beyond `fov_nm`
and cropping after the Gaussian filter, to get boundary-artifact-free
pixels at the FOV edges.
**Rationale:** `edge_amp_scale` is unavoidable -- the yield stage cannot run
without it, and CLAUDE.md's signature list reads as illustrative of the
key/non-obvious parameters, not a frozen contract (the geometry.py
`build_geometry` signature already grew similarly between steps 2 and 3).
Padding was deliberately **not** added: CLAUDE.md's recipe is exact and
literal ("supersample, apply gaussian_filter..., then
reshape(...).mean(...)") with no mention of margin handling, and adding it
now would be scope creep not requested for this step.
**Consequences:** Pixels within roughly one `psf_sigma_nm` of a rendered
FOV's outer edge carry a small reflect-mode boundary bias versus an
infinite-domain PSF. This should not matter for the interior region used
for localisation. Revisit with margin+crop if a later validation test
proves sensitive to edge-of-FOV pixels specifically (noted as an open
question, not a defect).
**Supersedes:** —

## D-015 — Dose controls Poisson SNR only, never brightness (scale up, Poisson, scale back down)
**Date:** 2026-08-16
**Status:** accepted
**Context:** `dose_e_per_px` must drive shot-noise SNR (validation test 2:
SNR proportional to sqrt(dose)), while CLAUDE.md's "Do not implement" list
explicitly excludes "calibrated dose->grey-level" -- dose must not also
determine image brightness.
**Decision:** `mean_electrons = yield * dose_e_per_px`; Poisson-sample at
that physical scale; then divide back by `dose_e_per_px` to return to a
yield-like value before converting to DN (`noisy_yield * _BASE_DN_PER_YIELD
* contrast_gain + brightness_offset`). `_BASE_DN_PER_YIELD = 100` is a
fixed, uncalibrated constant, independent of dose.
**Alternatives considered:** Applying Poisson directly to a DN-scale mean
(rejected: DN scale is arbitrary/uncalibrated, so Poisson's noise magnitude
-- which depends on the *absolute* value of its mean, not just its shape --
would be tied to that arbitrary DN scale rather than to physical dose,
breaking the dose->SNR relationship test 2 checks for).
**Rationale:** This is the only construction found that makes SNR scale
with `sqrt(dose)` (verified: `SNR/sqrt(dose)` roughly constant across
dose 50-5000) while keeping brightness independent of dose, matching both
CLAUDE.md's literal `gain/offset -> Poisson` ordering (the yield->electron
conversion via dose *is* the pre-Poisson "gain") and its explicit
prohibition on dose-calibrated grey levels.
**Consequences:** `contrast_gain`/`brightness_offset` are applied
*after* the Poisson+renormalise step (still before Gaussian read noise,
preserving the "Poisson then Gaussian" ordering) rather than literally
before Poisson -- a deliberate reading of "gain/offset" as the general
signal-formation stage, not a claim that every sub-field is pre-Poisson.
`_BASE_DN_PER_YIELD` is not calibrated to any real tool; do not present
absolute brightness from this generator as physically meaningful.
**Supersedes:** —

## D-016 — Row banding is Gaussian-correlated along rows (own engineering choice of correlation length)
**Date:** 2026-08-16
**Status:** accepted
**Context:** Validation test 5 checks that row-mean autocorrelation shows
"the injected length" -- banding needs *some* spatial correlation along the
row axis, not independent per-row noise (which would show no
characteristic length). `CaptureConfig.row_band_amp` gives an amplitude but
CLAUDE.md gives no correlation length.
**Decision:** Same construction as LER (D-006): white noise per row,
`gaussian_filter1d` along the row axis with `sigma=_BAND_CORR_LEN_PX=15`
(output pixels, not nm -- banding is a scan/detector artifact, not tied to
physical geometry), rescaled to the target `row_band_amp` std, broadcast
across all columns in a row.
**Alternatives considered:** Independent white noise per row (rejected: no
correlation length for test 5 to detect); a fixed sinusoidal banding
pattern (rejected: real scan artifacts are not perfectly periodic, and a
sinusoid has no single well-defined "length" either).
**Rationale:** Reuses an already-validated pattern (LER) rather than
inventing a new noise-correlation mechanism; 15 px is a plausible
scan-instability length scale and is easy to change if step 11's test 5
wants a different value to detect cleanly.
**Consequences:** `_BAND_CORR_LEN_PX` is an uncalibrated engineering choice
(c). Banding realisation is drawn from the capture's own noise RNG stream
(independent between reference and search, per constraint #4).
**Supersedes:** —

## D-017 — `quantise_8bit` selects the return dtype (uint8 vs clipped float64)
**Date:** 2026-08-16
**Status:** accepted
**Context:** CLAUDE.md's naming (`generate_pair(cfg) -> (ref_u8, search_u8,
truth)`) suggests uint8 output, but validation tests 6/7 need noiseless,
sub-DN-precision images for `phase_cross_correlation` to recover a 0.01 px
shift accurately -- 8-bit quantisation noise would corrupt that.
**Decision:** `apply_detector` returns `uint8` in `[0,255]` when
`cfg.quantise_8bit` is `True`, else `float64` clipped to `[0.0, 255.0]` (not
quantised). Clipping happens either way -- it's `quantise_8bit` specifically
that's optional, not the clip stage.
**Alternatives considered:** Always returning uint8 and hoping 8-bit
quantisation noise is small enough for tests 6/7 to still pass within
0.01 px (rejected as fragile and not what `CaptureConfig.quantise_8bit`
would otherwise be for -- the field exists precisely to be toggled).
**Rationale:** `ref_u8`/`search_u8` in CLAUDE.md's signature describes the
common/production case (`make_dataset.py` always uses
`quantise_8bit=True`), not a hard type constraint on `generate_pair`
itself -- the field being present and boolean implies it's meant to be
toggled, and tests 6/7 are the reason.
**Consequences:** Callers that need dtype certainty must check
`cfg.quantise_8bit`. `make_dataset.py` (step 10) must always set it `True`
so the shipped dataset is genuinely 8-bit.
**Supersedes:** —

## D-018 — Search FOV is anchored centered at world (0,0); only the reference is randomly placed within it
**Date:** 2026-08-16
**Status:** accepted
**Context:** `generate_pair` needs to decide where in world coordinates
both captures sit. `ref_center_nm=None` means "random placement" per
CLAUDE.md; there is no equivalent field for randomising the search
window's position.
**Decision:** `search_origin_nm` is always
`(-search_fov_nm/2, -search_fov_nm/2)` -- the search FOV centered at the
world origin. `ref_center_nm`, when `None`, is drawn uniformly at random
within the search FOV (minus a margin equal to half the reference FOV plus
50 nm, so the reference window is always fully contained).
**Alternatives considered:** Also randomising `search_origin_nm` per case
(e.g. representing "which part of a large virtual die we happened to
scan").
**Rationale:** The fin/gate lattice itself has no random phase (fin centers
are always exact multiples of `fin_pitch_nm` from the model's own origin --
per-case variety already comes from geometry's own LER/cut/landmark
realisation, which *is* freshly seeded every case, plus `ref_center_nm`'s
random placement giving lattice-phase diversity relative to the reference.
Randomising `search_origin_nm` too would add another free variable without
a clear benefit CLAUDE.md asks for, and would complicate reasoning about
`extent_nm` sizing (D-008) for no clear gain.
**Consequences:** Every case's search image is centered on the same nominal
lattice coordinates (just different LER/cut/landmark noise draws and a
different reference sub-position). If Phase 2 needs broader world-position
diversity, this is the parameter to revisit.
**Supersedes:** —

## D-019 — `subpixel_offset_px` rigidly shifts both `ref_center_nm` and `search_origin_nm` together
**Date:** 2026-08-16
**Status:** accepted
**Context:** Validation tests 6/7 (step 9/11) need two renders that differ
by a precisely controlled sub-search-pixel shift *in the rendered image
content*, recoverable via `phase_cross_correlation`. CLAUDE.md's own worked
numbers ("shift ref_center_nm by exactly 2 nm in x (= 0.2 search px)... must
return 0.200 +/- 0.01 px") only work out arithmetically if the compared
images are the two **search** captures (2 nm / 10 nm-per-search-px =
0.200) -- comparing two **reference** renders would give 2.0 px, not 0.200.
**Decision:** `subpixel_offset_px` (search-pixel units) converts to nm via
`search.pixel_size_nm` and is added to *both* `ref_center_nm` and
`search_origin_nm` after placement is resolved -- a rigid translation of
where the whole pair looks within the (shared, unchanged) geometry.
Because both shift equally, `gt_search_px` (the reference's position
*relative to* the search frame) is unchanged; only the search image's
*pixel content* shifts, by exactly the requested amount.
**Alternatives considered:** Interpreting `subpixel_offset_px` as shifting
only `ref_center_nm` (rejected: the search image would then be completely
unaffected by the offset, since the search window never moves -- there
would be nothing for `phase_cross_correlation` on the search images to
detect, and the arithmetic wouldn't match CLAUDE.md's `0.200 px` figure).
**Rationale:** This is the one reading that makes both the units (search
px, not reference px) and CLAUDE.md's exact worked numbers self-consistent.
**Consequences:** Tests 6/7 construct two `PairConfig`s with the same seed
and `ref_center_nm`, differing only in `subpixel_offset_px`, and run
`phase_cross_correlation` on the two **search** images (not reference).
**Supersedes:** —

## D-020 — Test 1's coarse-capture supersample raised to k=32 (was 8)
**Date:** 2026-08-16
**Status:** accepted
**Context:** Test 1 (physical correspondence) failed at 2.178 DN against a
2.0 DN threshold. A user-directed coordinate-system/subpixel-rendering
audit (see `docs/LOG.md`, "Step 9 follow-up") isolated the cause: test 1
uses a matched `psf_sigma_nm=1.0` at *both* the 1 nm/px and 10 nm/px
scales (not each capture's production default) to isolate pure
sampling/integration self-consistency. At the coarse (10 nm/px) scale with
the default `supersample=8`, the fine-pixel step is 1.25 nm, so
`sigma_fine_px = 1.0/1.25 = 0.8` -- the PSF's own standard deviation is
*smaller than one fine-grid sample spacing*, a severely under-resolved
discrete approximation of `gaussian_filter`'s continuous-domain target. A
k-sweep confirmed this is pure quadrature error, converging monotonically
toward zero: k=8 -> 2.178 DN, k=16 -> 0.791, k=32 -> 0.265, k=64 -> 0.097.
**Decision:** Raise `test_1_physical_correspondence`'s `coarse_cfg`
supersample from 8 to 32 nm. Nothing else in the test changed (geometry,
PSF sigma, detector config, ground truth, and the 2.0 DN threshold are all
untouched).
**Alternatives considered:** k=64 (more margin, ~4x more render cost for
another ~2.7x error reduction -- not needed since k=32 already clears the
threshold with >7x margin, 0.265 vs 2.0); k=16 (does clear the threshold,
0.791 vs 2.0, but with less margin than k=32 for a modest 2x cost saving).
**Rationale:** This is quadrature/discretization error, not a physics or
coordinate-convention bug (confirmed by k-dependence -- see the audit in
`docs/LOG.md`) and `supersample` is explicitly a configurable knob per
CLAUDE.md ("These must be configurable — validation test 7 may force them
higher"). Raising it is the spec-sanctioned remedy for a k-dependent
failure, distinct from tuning a threshold to force a pass.
**Consequences:** Test 1 passes (0.265 DN < 2.0 DN, confirmed by rerun).
Does not affect `optics.py`, `geometry.py`, or any other test's
configuration. Does not by itself say anything about tests 6/7's failure,
which was confirmed (same audit) to be k-*invariant* and therefore a
different phenomenon -- see D-021.
**Supersedes:** —

## D-021 — Search `psf_sigma_nm` raised from 3.0 to 8.0 nm (partial improvement, does not clear the gate)
**Date:** 2026-08-16
**Status:** accepted (interim -- tests 6/7 still fail at this value; see Consequences)
**Context:** Tests 6/7 fail with a k-invariant (not discretization-driven)
subpixel-recovery bias, isolated by the same audit (`docs/LOG.md`) to
genuine spatial aliasing: `se_yield`'s deliberate edge discontinuity
(D-012), blurred by a PSF narrow relative to the 10 nm search pixel, still
carries real spectral content above the pixel's Nyquist frequency; box-
averaging (decimating) that content to the pixel grid is not
shift-equivariant. CLAUDE.md's own contingency for exactly this failure
mode is "reconsider σ_beam for the search capture." The user authorized a
bounded, boundary-controlled sweep at sigma in {3, 5, 8} nm (not an
unbounded search, and explicitly not "pick whichever passes test 7") to
choose the smallest physically defensible value giving stable behavior
with no boundary artifact.
**Decision:** Set the search capture's `psf_sigma_nm` to **8.0 nm** (was
3.0 nm) for tests 6/7 (`generator/validate.py::_SEARCH_PSF_SIGMA_NM`).
Sweep results (isolated single edge, no LER/cuts/periodicity, small FOV
2000 nm vs. 6x-larger 12000 nm FOV; real intel14 lattice with a 500 nm
extent margin beyond the usual 100 nm):
| sigma | isolated-edge slope | isolated-edge residual ptp | real-lattice slope | real-lattice residual ptp | boundary effect |
|---|---|---|---|---|---|
| 3 nm | 1.130 | 0.461 | 1.101 | 0.117 | none detected |
| 5 nm | 1.092 | 0.346 | 1.056 | 0.064 | none detected |
| 8 nm | 1.055 | 0.252 | 0.968 | 0.048 | none detected |
Zero measurable difference between the small and 6x-larger FOV at any
tested sigma (identical to 4 decimal places), and a separate sigma=3,
FOV-only sweep (2000/4000/12000 nm) also showed zero change -- ruling out
`optics.py`'s documented reflect-mode FOV-boundary bias (D-014) as a
contributor at any of these sigma values. The trend is monotonically
improving across 3->5->8 nm on both geometries, with no sign of the
non-monotonic "worse at higher sigma" shape seen in an earlier, less
controlled sweep (that earlier result is not reproduced here and is judged
to have been an artifact of a narrower, less-margined test setup, since it
is not reproduced under a boundary-controlled comparison).
**Alternatives considered:** Keep 3.0 nm (rejected: strictly worse than 8
nm on every metric tested, no offsetting benefit found -- the "narrower is
more physically literal" argument is already moot since 3.0 nm was itself
already an assumption-driven extension beyond the literal 0.85-1.7 nm
probe-derived range, not a hard physical floor). Sweeping past 8 nm to
find a value that fully clears the gate (explicitly out of scope for this
pass -- the user capped the sweep at {3,5,8} and asked not to pick based on
which value passes).
**Rationale:** 8 nm is the best of the three tested values on every metric,
with no evidence of a countervailing boundary-artifact cost, so it is the
correct choice within the authorized sweep even though it does not fully
resolve the failure.
**Consequences:** Rerunning the actual tests (not the quick sweep) at
sigma=8 nm gives: test 6 recovers 0.150 px (was 0.130 px) against
0.200 +/- 0.01 -- **still fails**, though the gap shrank from 0.070 to
0.050. Test 7: slope 1.0151 (was 1.1247) -- **now passes** the 1.00 +/- 0.02
slope criterion -- but residual peak-to-peak is 0.0868 px (was 0.1274) --
**still fails** the < 0.025 px no-staircase criterion, though nearly
halved. The trend across the sweep had not plateaued by 8 nm, so a higher
sigma would likely continue to help, but that is future work, not decided
here. **Tests 6 and 7 remain failing.** This is reported, not hidden or
threshold-tuned around, per CLAUDE.md's explicit instruction. Whether to
extend the sigma search further, accept a different failure mode, or
revisit the validation criterion itself (CLAUDE.md's own option "C") is a
decision for the user, not made unilaterally here.
**Supersedes:** Informally revises the literal CLAUDE.md default of
`psf_sigma_nm = 3.0` for the search capture; no prior D-entry had formally
adopted that value, so nothing is being reversed, only extended.

## D-022 — Production coarse/search supersample set to k=16 (not k=32); k=32/64 kept as convergence-validation references only
**Date:** 2026-08-16
**Status:** accepted
**Context:** D-020 fixed test 1 by raising its coarse-capture supersample
to 32, chosen for margin (0.265 DN vs. the 2.0 DN threshold) without
weighing render cost. The user pointed out k=16 already passes
(0.791 DN < 2.0 DN, ~2.5x margin) at roughly half the render cost of k=32,
and asked for a single named "production" coarse/search supersample
constant, with k=32/k=64 demoted to on-demand convergence-check references
rather than the value tests actually run at.
**Decision:** Introduced `validate.py::_PRODUCTION_SEARCH_SUPERSAMPLE = 16`
and used it for both test 1's `coarse_cfg` (was 32) and tests 6/7's
`search_cfg` (was 8, the CaptureConfig/CLAUDE.md-literal default). k=32 and
k=64 remain documented as the convergence-validation points (D-020's
k=8/16/32/64 sweep: 2.178 / 0.791 / 0.265 / 0.097 DN) that establish k=16
sits on the converged part of the curve, not a lucky plateau -- they are
not wired into any test's default configuration.
**Alternatives considered:** Leaving test 1 at k=32 (more margin, but 2x
the render cost of k=16 for no requirement that needs it -- the 2.0 DN
threshold only needs ~2.5x margin, which k=16 already gives). Leaving
tests 6/7's `search_cfg` at k=8 (also valid, since k=8/16/32 were already
shown k-invariant for that failure mode -- see the Step 9 audit in
`docs/LOG.md`) -- changed anyway for a single consistent "production
coarse/search supersample" value across the suite rather than two
different supersample constants with no functional reason to differ.
**Rationale:** Supersample is a pure numerical-accuracy knob (not a
physics parameter); once a value is shown to be on the converged part of
the quadrature-error curve, using the cheapest such value is the correct
default, and 32/64 remain valuable as periodic confirmation that k=16 has
not silently drifted off the converged plateau (e.g. if `fin_width_nm` or
`psf_sigma_nm` changes for some future preset in a way that re-narrows the
effective resolution margin).
**Consequences:** Reran all three gate tests at k=16.
- Test 1: **passes**, 0.7906 DN < 2.0 DN (matches the D-020 k=16 sweep
  prediction of 0.791 DN almost exactly).
- Test 6: recovered 0.150 px vs. 0.200 +/- 0.01 -- **still fails**,
  essentially identical to the k=8 result (0.150), as expected from D-021's
  established k-invariance for this failure mode.
- Test 7: slope 1.0122 (vs. 1.0151 at k=8, both pass 1.00 +/- 0.02);
  residual peak-to-peak 0.0755 px (vs. 0.0868 at k=8) -- **still fails**
  the < 0.025 px criterion. The small k=8 -> k=16 numeric differences here
  are within the noise expected from re-discretizing at a different fine
  grid, not a meaningful trend -- consistent with, not contradicting,
  tests 6/7's established k-invariance.
No threshold, PSF sigma, geometry, or detector parameter changed in this
step -- purely a supersample/render-cost decision.
**Supersedes:** D-020's choice of k=32 as test 1's coarse-capture
supersample (k=32 is retained as a documented convergence-reference point,
not reversed as a finding -- only demoted from "the value tests run at" to
"a value used to validate the production choice").

## D-023 — `presets.py` L0-L6 ladder: search σ=8.0nm/k=16 as the production baseline; L4's PSF-divergence step translated onto the new baseline
**Date:** 2026-08-16
**Status:** accepted
**Context:** Building the L0-L6 difficulty ladder (Step 12) requires a
baseline `psf_sigma_nm`/`supersample` for reference and search captures.
CLAUDE.md's literal defaults (search `psf_sigma_nm=3.0`, supersample
reference=4/search=8) predate D-020/D-021/D-022's evidence-based revision
to search `psf_sigma_nm=8.0` and supersample=16 for both captures. The
user directed that these revised production values (σ_search=8nm, k=16)
be carried forward into the benchmark presets and dataset generation, with
the residual subpixel bias documented as an accepted Phase 1 limitation
rather than chased further or hidden by a threshold change.
**Decision:** `presets.py` uses `psf_sigma_nm=1.0` (reference) /
`8.0` (search) and `supersample=16` (both) as the L0-L3 baseline. CLAUDE.md's
L4 rung explicitly widens search PSF as an added difficulty axis (literal
spec: search 3.0 -> 3.5 nm, a +0.5 nm step); rather than use the stale
absolute value 3.5 nm (which would make L4 *narrower*, hence per our own
sweep *more* subpixel-aliased, than the L0-L3 baseline -- inverting the
ladder's intended monotonic difficulty), L4-L6 apply the same *relative*
+0.5 nm step to the new baseline: search `psf_sigma_nm=8.5` nm.
Reference's L4 PSF stays at the literal spec value (1.0 nm, unchanged from
baseline). `edge_amp_scale` asymmetry at L4 (CLAUDE.md: "differing
0.9/1.1", no capture assignment given) is assigned reference=0.9,
search=1.1 -- an arbitrary but documented choice; Phase 2 is not expected
to know a priori which capture received which multiplier, so the
assignment direction should not matter for benchmark validity.
**Alternatives considered:** Using CLAUDE.md's literal 3.0/3.5 nm search
PSF values throughout `presets.py`, decoupled entirely from the validate.py
gate's revised 8.0 nm (rejected: would ship a dataset with *worse*,
already-documented-as-suboptimal subpixel behavior than what the gate
investigation established was achievable, undermining the point of having
done that investigation). Scaling L4's step proportionally
(3.5/3.0 x 8.0 ≈ 9.33 nm) instead of additively (rejected: no more
principled than the additive version, and less directly traceable to
CLAUDE.md's literal delta).
**Rationale:** The gate investigation (D-021) was explicitly framed as
establishing search σ as a project-wide production parameter, not a
validate.py-only special case; the user's own instruction sequencing
("keep σ_search=8nm ... complete the benchmark presets") confirms this
reading. Preserving CLAUDE.md's *relative* ladder structure (rather than
its now-stale absolute numbers) keeps the difficulty-escalation intent
intact without re-litigating the psf_sigma_nm=8.0 decision.
**Consequences:** L0-L3 search images carry the same documented residual
subpixel bias characterised in D-021 (test 6: ~0.15 px recovered for a
true 0.20 px shift; test 7: slope ~1.01, residual ptp ~0.075 px against a
0.025 px threshold) -- this is a known, accepted Phase 1 limitation of the
shipped dataset, not something Phase 2 should be surprised by. L4-L6 add a
further, deliberate PSF mismatch on top of that baseline as an explicit
difficulty axis. If a future revision changes the production search sigma
again, this file's `_SEARCH_PSF_SIGMA_NM`/`_SEARCH_PSF_SIGMA_NM_L4PLUS`
constants are the single place to update.
**Supersedes:** —

## D-024 — Shipped dataset uses supersample k=8, not the validation gate's k=16 (memory-bounded, not a numerics reversal)
**Date:** 2026-08-16
**Status:** accepted
**Context:** While timing production-scale renders for `make_dataset.py`
(Step 10), `render()` at CLAUDE.md's required full 1000x1000 px output and
`supersample=16` (D-022's production value) raised
`numpy._core._exceptions._ArrayMemoryError` -- the fine grid is
`out_px * k` per side, so 1000 px at k=16 is a 16000x16000 grid; a single
float64 array at that size is 1.9 GiB, and `signed_distance()` holds many
such arrays simultaneously mid-computation (fin/gate distance terms, LER
offsets, cut terms, landmark loop), needing an estimated 10-15+ GiB peak --
more than the ~10 GB available in this environment. This was not caught
earlier because every prior use of k=16 (the gate tests) rendered at
100-200 px (1600-3200 px fine grid), a small fraction of the size.
**Decision:** `presets.py` uses `supersample=8` (not 16) for every L0-L6
level and the cut_density sweep -- i.e., for everything that
`make_dataset.py` actually ships to `data/`. `validate.py`'s gate tests
(1/6/7) are unchanged and continue to use k=16 (or k=32/64 as convergence
references) at their existing small image sizes, where it is both
numerically justified and memory-feasible.
**Alternatives considered:** Implementing chunked/tiled rendering in
`optics.py::render()` (splitting the fine grid into row-strips with
overlap margins for the Gaussian filter, keeping peak memory bounded) so
k=16 could genuinely run at 1000 px -- more faithful to a literal "k=16
everywhere" reading, but is new engineering work on a core function three
gate tests already depend on, with real risk of subtly changing behavior
under time pressure (would require re-verifying tests 1/6/7 afterward).
Shipping datasets below the CLAUDE.md-required 1000x1000 px resolution so
k=16 fits in memory -- rejected outright: deviates from a non-negotiable
spec requirement, a larger problem than the supersample question. The user
was presented with these three options and chose k=8 for the shipped
dataset.
**Rationale:** k=8 is not merely "the only option that fits" -- it is
independently confirmed numerically adequate for every psf/pixel
combination the L0-L6 presets actually use:
`sigma_fine_px = psf_sigma_nm * k / pixel_size_nm` gives 1.0*8/1.0 = 8 for
the reference capture and 8.0*8/10.0 = 6.4 for search, both comfortably
above the ~1 threshold where D-020's k-sweep found quadrature error become
significant. That under-resolution was specific to test 1's contrived
matched-`psf_sigma_nm=1.0`-at-10-nm/px configuration (`sigma_fine_px=0.8`
at k=8) -- a scenario no L0-L6 preset actually uses (search's own
psf_sigma_nm is 8.0/8.5 nm throughout the ladder, never the pathological
1.0 nm test 1 deliberately used to isolate quadrature error). k=16 remains
the right choice where it is both needed and affordable (the gate tests);
this is a scale-dependent practical constraint, not a reversal of
D-020/D-022's numerical finding.
**Consequences:** ~185 s of render time per pair (103 s reference + 82 s
search, empirically timed) at k=8/1000px, before detector noise and file
I/O -- `make_dataset.py`'s case count was sized with this cost in mind
(see its own docstring/case-count choice). If a future preset introduces a
psf_sigma_nm/pixel_size_nm ratio that pushes `sigma_fine_px` back toward
the ~1 danger zone at k=8, this decision must be revisited (either a
higher k for that specific preset if it fits in memory, or the tiled-
rendering alternative above).
**Supersedes:** —

## D-025 — Tests 3/4 isolated-edge fixture: widened synthetic geometry (`preset="custom"`), corrected window sizing, and peak-to-trough width metric for test 3
**Date:** 2026-08-17
**Status:** accepted
**Context:** Steps 10-12 were complete and the outstanding gap was tests
3/4/5/8/9. Running them together first hung indefinitely: tests 3 and 4
render at `out_px=400` over a 40 nm FOV (0.1 nm/px) with `k=16`
supersampling, giving `fine_step_nm=0.00625`; for `psf_sigma_nm` up to
4.0 that is `sigma_fine_px` up to 640, and `scipy.ndimage.gaussian_filter`'s
cost scales with kernel radius (`truncate*sigma`), so the sweep was
convolving with a ~2500-pixel kernel over a 6400x6400 array -- confirmed
via `Get-Process` showing the worker steadily accumulating CPU time
(not deadlocked, just computationally infeasible; ~150s once fixed vs.
>20 min unfinished before). Lowering `k` to 4 (fine step 0.025 nm, still
far finer than any swept sigma) fixed the hang with no accuracy cost.
Fixing the hang then exposed two independent, genuine test-fixture bugs
that the hang had been masking:
(1) The real intel14 fin is 8 nm wide, so the edge under test's own
opposite edge sits only 8 nm away -- inside the 3*sigma=12 nm influence
radius of the largest swept sigma (4 nm). The window's own "10 nm into
the fin interior" margin actually overshot past that opposite edge
entirely. This produced non-monotonic, even zero, measured widths -- not
a renderer bug, a real physical consequence of probing an 8 nm-wide
feature with a beam approaching that same scale, using a fixture that
assumed isolation it didn't have.
(2) Even after fixing (1) by widening the window, test 3 still failed,
now with widths shrinking smoothly and monotonically with sigma. Direct
numeric inspection (`profile[out_px//2,:]` for sigma in
{0.5,1.0,2.0,4.0}) showed why: `Y_SUBSTRATE == Y_FEATURE == 0.817`, so
this yield model has zero bulk contrast -- the entire visible signal is
a transient bright peak just inside the edge (from ALPHA_E/ALPHA_V) and
a dark trough just outside it (from PHI_F/PHI_B). `profile.min()`/`max()`
are exactly that peak and trough, and PSF blur reduces their *amplitude*
toward the flat baseline even while correctly increasing their spatial
*separation* (measured directly: peak-trough dx = 2.10, 3.70, 6.70,
12.50 nm at sigma = 0.5, 1.0, 2.0, 4.0 nm -- cleanly monotonic). A
20-80%-of-(max-min) crossing distance is therefore not a valid "edge
broadened" indicator for this specific yield model shape, regardless of
window sizing.
**Decision:** Three changes, all in `validate.py` only (no change to
`yield_field.py`/`optics.py`/`detector.py` -- this is a test-measurement
fix, not a forward-model change): (a) `_isolated_edge_model()` now
builds geometry with a new `preset="custom"` escape hatch in
`config.resolve_preset()` (returns `geometry` unchanged when
`preset=="custom"`, bypassing the `NODE_PRESETS` lookup) with
`fin_pitch_nm=250, fin_width_nm=60` -- purely to give ~50 nm clearance
to the nearest other edge, far beyond any swept sigma's influence;
`fin_height_nm`/`gate_pitch_nm`/`gate_length_nm` stay at intel14's real
values so the fin-height-dependent backscatter term stays representative.
This fixture is renamed `_ISOLATED_EDGE_X_NM` and is separate from test
9's `_EDGE_TEST_X_NM=46.0`, which still probes the real intel14 preset
directly (test 9 needs the actual preset's LER realisation, not an
isolation fixture -- reusing one constant for both was itself a bug this
change introduced and then fixed within the same session). (b) The
window margins were widened to 20 nm interior / 30 nm substrate. (c)
Test 3's width metric changed from a 20-80%-of-(profile.min(),
profile.max()) crossing distance to the peak-to-trough x-separation
directly -- the metric a real CD-SEM edge-width measurement would use on
a linescan with this bright-peak/dark-trough shape. Test 4 keeps its
original peak-vs-plateau (`profile[-20:].mean()`) ratio metric, which
was already immune to this failure mode by construction (D-012 already
documented that test 4 deliberately does not use the raw analytic peak),
and passed as soon as the fixture isolation bug (1) was fixed.
**Alternatives considered:** (i) Keep the real intel14 fin and shrink the
swept sigma range so 8 nm stays enough clearance -- rejected, this is
literally the "tune the threshold/range to make it pass" move CLAUDE.md
prohibits, and CLAUDE.md's own test-3 spec fixes the sweep at 0.5-4 nm.
(ii) Define plateau references from fixed-offset sub-windows (e.g. mean
of the first/last 5 nm) instead of profile.min()/max() -- rejected after
checking numerically: because `Y_SUBSTRATE==Y_FEATURE`, both fixed
references converge to the same ~0.82 baseline, making the normalization
range degenerate (near zero) rather than merely wrong direction. (iii)
Add a brand-new named process-node preset with a wider fin -- rejected;
that would misrepresent a test-only fixture as a real node option in
`presets.py`'s public NODE_PRESETS table, which CLAUDE.md restricts to
sourced values for the three named nodes.
**Rationale:** All three changes are fixes to how the test measures an
already-correct forward model, not adjustments to any physics constant,
threshold, or forward-model ordering. The two-stage discovery (isolation
bug, then metric bug) is preserved above because each was independently
diagnosed with direct numeric evidence rather than by tuning until green,
per CLAUDE.md's explicit prohibition on threshold-tuning.
**Consequences:** `preset="custom"` is now a general capability of
`resolve_preset()`, not just a test hack -- it is unused by any
production preset in `presets.py` and should stay that way unless a
future need for a genuinely bespoke (non-node) geometry arises.
Documented edge-width numbers for the production intel14 geometry itself
were not computed here (only the widened test fixture) since CLAUDE.md's
test 3 asks about the PSF's own broadening behavior in isolation, not
about intel14's specific fin width -- the two are the same effect, this
fixture just removes the 8 nm fin's self-interference from the
measurement.
**Supersedes:** —
