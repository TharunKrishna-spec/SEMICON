# Build Log

Append-only. Chronological, one block per work session or per completed step.

### 2026-08-16 — Step 0: repo skeleton + docs
- Created `generator/`, `suites/`, `docs/` (`DECISIONS.md`, `LOG.md`,
  `sources.md`, `assumptions.md`, `validation/`), `data/`, `truth/`.
- Seeded `docs/sources.md` with every constant given in CLAUDE.md, tagged
  (a)/(b)/(c).
- Initialised a git repository at the project root.
- Next: config dataclasses.

### 2026-08-16 — Step 1: config dataclasses
- Implemented `GeometryConfig`, `CaptureConfig`, `PairConfig` exactly as
  specified in CLAUDE.md.
- Implemented `resolve_preset()` using a `NODE_PRESETS` dict keyed by preset
  name; returns a copy via `dataclasses.replace` rather than mutating the
  input.
- Verified: instantiated each of `intel14`/`n7`/`n5`, confirmed resolved
  field values match the table in CLAUDE.md; confirmed an unknown preset
  name raises `ValueError`.
- Next: geometry + signed distance.

### 2026-08-16 — Step 2: geometry + SDF
- Implemented fins (vertical stripes, width `fin_width_nm`, pitch
  `fin_pitch_nm`) and gates (horizontal stripes, width `gate_length_nm`,
  pitch `gate_pitch_nm`) as an index-based (round-to-nearest-pitch-multiple)
  signed distance rather than an explicit rectangle list — see D-001.
  Union via `np.maximum` (positive-inside convention) — see D-002.
- Whole-lattice rotation (`rotation_deg`) applied once to query points about
  the origin, not per-rectangle, so fins/gates stay parallel instead of
  fanning out — see D-003. Confirmed by rendering `rotation_deg=0` and `2`
  side by side in `docs/validation/step2_sdf.png`.
- Verified by three independent checks: (1) point probes at known
  fin-center / gate-center / crossing / substrate locations match
  hand-computed values exactly (e.g. crossing at origin: `d=10.0` =
  `max(fin_half_w=4, gate_half_w=10)`); (2) 1D cross-sections — along a gate
  centerline `d` is a constant `+10`; through a substrate row (no gate) `d`
  is a triangle wave between `+4` and `-17` with period = `fin_pitch_nm`;
  along a fin centerline `d` is `+4` baseline with `+10` peaks at
  period = `gate_pitch_nm`. All match the closed-form expectation exactly.
  (3) rendered `docs/validation/step2_sdf.png` and eyeballed the 2D field.
- Gotcha: the first pass at the 2D plot used a diverging colormap
  (`coolwarm`) whose sign mapping was easy to misread at a glance — I
  initially read the fin/gate regions as blue (mentally: "blue = inside
  feature") when for this `vmin`/`vmax` choice red is actually the high
  (inside-feature) end. Caught it by cross-checking against the point
  probes and 1D slices rather than trusting the thumbnail; relabelled the
  colorbar (`red = inside feature, blue = substrate`). This was a
  plotting/reading mistake, not a bug in `geometry.py` — logging it because
  "the render looked wrong" is exactly the kind of dead-end that's cheap to
  record and expensive to rediscover.
- Not yet implemented (by design, per build order): corner rounding
  (`corner_radius_nm` unused, fins/gates are sharp-cornered — see D-004),
  LER, cuts, landmarks.
- Open: corner rounding, LER, cuts, landmarks.
- Next: Step 3, LER.

### 2026-08-16 — Step 3: line-edge roughness (LER)
- Added `_EdgeRoughness`: white noise on a 2 nm grid, smoothed with
  `scipy.ndimage.gaussian_filter1d` (kernel std = `ler_corr_len_nm`),
  rescaled to hit the target `ler_sigma3_nm / 3` std exactly — see D-006.
  Four independent realisations per model: fin-left, fin-right, gate-bottom,
  gate-top (edges not width-correlated — see D-007).
- `build_geometry()` signature changed: now takes `extent_nm` (half-width of
  the square domain LER is realised over) and `rng` (a dedicated geometry
  RNG stream, separate from per-capture noise streams) — see D-008. This is
  a real API change from step 2, expected at this point in the build order.
- `signed_distance` now perturbs each of the four edge positions by the
  corresponding `_EdgeRoughness.sample(feature_idx, running_coord)` before
  taking `min(dist_to_edge_a, dist_to_edge_b)` per feature and
  `max(d_fin, d_gate)` for the union, same structure as step 2. Confirmed
  algebraically that at zero roughness this reduces to exactly the step-2
  formula (`half_w - |offset from center|`).
- Verified: (1) determinism — same seed twice gives bit-identical
  `_EdgeRoughness.values` arrays; different seed gives different arrays;
  (2) realised std of each of the four arrays matches the target
  `ler_sigma3_nm/3 = 1.0 nm` exactly (post-rescale, as designed);
  (3) `signed_distance` stays finite everywhere on a 500x500 test grid;
  (4) rendered `docs/validation/step3_sdf_ler.png`, `ler_sigma3_nm=0` vs
  default (`3.0`/`25.0`) side by side — LER off shows perfectly straight
  edges, LER on shows smooth correlated wiggle at roughly the expected
  spatial scale relative to the 200 nm field of view shown, on both fin and
  gate edges.
- Open: corner rounding, cuts, landmarks. `extent_nm` sizing for real 10 um
  search captures is deferred to `pair.py` (step 8).
- Next: Step 4, cuts and landmarks.

### 2026-08-16 — Step 4: cuts and landmarks
- Cuts: candidate sites at gap midpoints between perpendicular-feature
  crossings (fin-cut sites between gate rows, gate-cut sites between fin
  columns), independent per-site Bernoulli draw at `cut_density` — see
  D-009. Cut length = 30% of the perpendicular pitch — see D-010. Applied
  via CSG subtraction (`min(d_feature, -notch_sdf)`, `+inf` where no cut is
  present at the nearest site so it's a no-op there), same "exact away from
  corners" approximation class as the fin/gate union (D-002).
  Double-checked the subtraction sign convention against a hand worked
  example (a query point just past a cut boundary but still centered in a
  fin) — initially thought the formula was leaking distance from the cut
  into unrelated fin territory, then re-derived by hand and confirmed the
  reduced distance is *correct*: the cut boundary genuinely is the nearest
  edge there. No bug, but worth the sanity check before trusting it.
- Landmarks: axis-aligned square pads (exact box SDF), `landmark_scale_nm`
  side, `landmark_count` of them, centers uniform-random within 80% of
  `extent_nm`, no LER on landmark edges — see D-011. Unioned in via the same
  `max()` as fins/gates.
- Refactored the fin/gate index-range arithmetic (previously inlined twice
  in `_make_edge_roughness_pair`) into a shared `_index_range()` helper, now
  reused for LER rows, LER sample columns, and cut feature/site indices.
- Verified: (1) `cut_density=0, landmark_count=0` stays finite and matches
  the step-3 value range (no regression); (2) determinism — same seed twice
  gives identical `fin_cuts.present` / `gate_cuts.present` /
  `landmark_centers_nm` and identical `signed_distance` output; (3) realised
  cut fraction at `cut_density=0.3` came out ~0.30/0.28 for fins/gates,
  matching the target within sampling noise; (4) rendered
  `docs/validation/step4_cuts_landmarks.png` — left panel (`cut_density=0.35`,
  elevated for visibility) shows fins/gates visibly severed at scattered
  gaps, breaking the periodic look; right panel (`landmark_count=3`,
  `scale=150nm`) shows three large clean square pads sitting distinctly
  among the regular texture at their reported centers.
- Not yet implemented: corner rounding (`corner_radius_nm` still unused).
- Open: corner rounding.
- Next: `yield_field.py` (Mack-Bunday two-exponential SE yield model).

### 2026-08-16 — Step 5: SE yield field (Mack-Bunday ALM)
- Implemented `se_yield(d_nm, fin_height_nm, edge_amp_scale=1.0)` exactly per
  CLAUDE.md's formula, no extra physics. Applicability caveat (Si-on-Si,
  500 eV, 90-deg sidewall, 20-100 nm feature height) recorded as an
  `# ASSUMPTION:` comment directly above the constants.
- `edge_amp_scale` scales `ALPHA_E`/`ALPHA_V`/`PHI_F`/`PHI_B` together,
  leaving the flat baseline untouched — see D-013.
- Gotcha: the formula as literally specified is discontinuous at `d=0`
  (outside branch gives `0.352`, inside branch's `d->0+` limit gives
  `1.827`). First reaction was to suspect a spec inconsistency, but worked
  through it by hand: this is the expected "volume-loss dip right at the
  edge, enhancement peak just inside" shape of a real edge-yield response,
  and it's the beam PSF (step 6) that's supposed to smooth it into the
  familiar bright-edge CD-SEM look — not something to fix here. See D-012.
  Kept the formula exactly as given.
- Gotcha: `np.where` evaluates both branches over the whole array before
  selecting, so very negative `d` overflows `exp(-d/SIGMA_V)` (fast, 0.26 nm
  decay) in the branch that gets discarded anyway — confirmed via `np.where`
  semantics that overflow/NaN in the unselected branch never leaks into the
  output (checked explicitly, including the `edge_amp_scale=0` case where
  `0 * inf = NaN` could otherwise be a trap), then silenced the resulting
  `RuntimeWarning` with a scoped `np.errstate` rather than leaving noisy
  warnings that could mask a real problem later. This never occurs in
  practice from real geometry queries (`d` is bounded by pitch/landmark
  scale, well within safe range) — only showed up in a synthetic far-field
  convergence check.
- Verified: far-field plateaus converge to `Y_SUBSTRATE == Y_FEATURE ==
  0.817` (substrate side converges slowly — `LAMBDA_B` scales with fin
  height, ~34 nm for `intel14`, full convergence needs several hundred nm);
  boundary values match hand computation exactly (`0.352` / `1.827`); true
  peak `~2.10` at `d~0.4 nm`; `edge_amp_scale=0` gives a flat `0.817`
  everywhere; excursion above/below baseline scales linearly with
  `edge_amp_scale` (checked ratio at `d=1`: exactly `2.0x` for
  `edge_amp_scale=2` vs `1`). Rendered
  `docs/validation/step5_yield.png`: (1) annotated 1D `Y(d)` profile,
  (2) `edge_amp_scale` sweep, (3) full 2D yield map from the Step 4
  geometry model (LER + cuts + landmarks) — already looks like a
  recognisable bright-edge SEM image even before any PSF/noise is applied.
- Next: `optics.py` (Gaussian PSF + supersample/area-integrate).

### 2026-08-16 — Step 6: optics.py (Gaussian PSF + pixel-area integration)
- Implemented `render(model, origin_nm, fov_nm, out_px, psf_sigma_nm, k,
  edge_amp_scale=1.0)` exactly per CLAUDE.md's recipe: supersample to a fine
  grid, `signed_distance` -> `se_yield` at the fine grid, `gaussian_filter`
  with sigma converted to fine-pixel units, then
  `reshape(out_px,k,out_px,k).mean(axis=(1,3))` for exact pixel-area
  integration (not a resize). `edge_amp_scale` added beyond the literal
  signature (unavoidable — the yield stage needs it) and no FOV padding
  around the Gaussian filter (literal recipe as given) — see D-014.
- Gotcha (performance): a naive validation call at full production scale
  (1000 px out, k=4, i.e. a 4000x4000 fine grid) took ~18s wall time in this
  dev environment — profiled it stage by stage rather than guessing:
  `signed_distance` ~11s, `se_yield` ~3s, `gaussian_filter` ~3s. No
  accidental per-pixel Python loop or redundant recomputation found — this
  is the genuine cost of vectorised NumPy ops at 16M elements in this
  sandboxed environment, not an algorithmic bug. Left as-is per this step's
  explicit scope (no optimisation requested); flagging for step 10
  (`make_dataset.py`) runtime budgeting — at ~15-20s/capture x 2
  captures x >=30 cases, dataset generation is a "~15-20 minute batch job,"
  not interactive, which should be fine but is worth knowing going in.
- Gotcha (test methodology, not a code bug): first attempt at the PSF
  edge-broadening check used the real `intel14` lattice directly and got
  inconsistent/non-monotonic results at large sigma. Root-caused it: the
  34 nm fin-to-fin gap is narrow enough that at `psf_sigma_nm` up to 4 the
  measurement window either clipped the true peak/dip (window too narrow)
  or picked up the *next* fin's edge entirely (window too wide) —
  `signed_distance`'s `min(dist_left, dist_right)` structurally hands off
  which edge governs a point exactly at the inter-fin midpoint, so a wide
  enough window silently starts measuring a second, different edge. Fixed
  by building an isolated-single-edge test geometry directly (bypassing
  `resolve_preset`'s preset-overwrite via the private `_make_*` helpers,
  `fin_pitch_nm=1000`) so there is only one edge in view, no matter the
  window size. This is a test-harness fix, not a change to `optics.py` or
  `geometry.py`.
- Verified: with the isolated-edge geometry, the 20-80% edge-transition
  width increases strictly monotonically across `psf_sigma_nm` = 0.5 -> 4.0
  nm (0.73 nm -> 4.75 nm, every step increasing — see
  `docs/validation/step6_psf_broadening.png`). This is an informal
  precursor to formal validation test 3 (step 11), which will need its own
  more careful design given the narrow-gap edge case just found.
- Next: `detector.py` (gain/offset -> Poisson -> Gaussian read -> banding ->
  clip -> uint8), no detector noise added yet in this step per explicit
  scope.

### 2026-08-16 — Step 7: detector.py (gain, Poisson, read noise, banding, quantise)
- Implemented `apply_detector(yield_image, cfg, rng)`: yield -> electrons
  via `dose_e_per_px` -> Poisson -> renormalise back to yield scale -> DN via
  fixed `_BASE_DN_PER_YIELD=100` + `contrast_gain` + `brightness_offset` ->
  Gaussian read noise -> row banding (Gaussian-correlated, reusing the LER
  construction) -> clip `[0,255]` -> optional uint8 quantise. See D-015
  (dose/SNR-not-brightness), D-016 (banding correlation length), D-017
  (`quantise_8bit` toggles return dtype for the noiseless precision tests).
- Verified: dose->SNR roughly `SNR/sqrt(dose)` constant across dose
  50-5000 (30-trial Monte Carlo per dose point — noisy but the intended
  trend is clearly present; a tighter check belongs in the formal test 2 at
  step 11). Determinism (same seed -> bit-identical). `quantise_8bit`
  toggles `uint8` vs `float64` correctly. Banding row-structure: with dose
  set very high and read noise off to isolate it, row-mean std matched
  `row_band_amp` almost exactly (4.9999... vs target 5.0) while column-mean
  std was ~0.0002 (flat) — matches test 5's expected qualitative shape.
- Next: `pair.py` (RNG stream splitting, shared geometry, both renders,
  truth dict) — the integration step tying geometry/optics/detector
  together per capture.

### 2026-08-16 — Step 8: pair.py (generate_pair, shared geometry, truth)
- Refactored `geometry.py`: extracted the rotation-into-local-frame math
  from inside `signed_distance` into a new public `to_local_frame(model, X,
  Y)`, since `pair.py`'s lattice-phase/landmark-distance diagnostics need
  the exact same transform. `signed_distance` now calls it too — no
  behaviour change, confirmed by a direct before/after call.
- Implemented `generate_pair(cfg) -> (ref_image, search_image, truth)`:
  `np.random.default_rng(cfg.seed).spawn(4)` gives `geom_rng` (shared,
  passed to the one `build_geometry` call used for both `render()` calls —
  structurally impossible for the two captures to see different geometry),
  `placement_rng` (reference random placement when `ref_center_nm is
  None`), and independent `ref_noise_rng`/`search_noise_rng` for
  `apply_detector`.
- Search FOV anchored centered at world (0,0); reference placed randomly
  inside it with margin, or validated to fit if given explicitly — see
  D-018. `subpixel_offset_px` (search-px units) rigidly shifts both
  `ref_center_nm` and `search_origin_nm` together — worked out from
  CLAUDE.md's own numeric example (2 nm = 0.200 search px only makes sense
  if tests 6/7 compare the two **search** images, not reference) — see
  D-019.
- Truth dict: `ref_center_nm`, `search_origin_nm`, `gt_search_px` (float,
  search-px), `lattice_phase_fin`/`lattice_phase_gate` (signed fraction of
  one pitch, `[-0.5, 0.5)`, via `to_local_frame`), `dist_to_nearest_landmark_nm`
  (`None` when `landmark_count=0`).
- Verified: moderate-scale (200px) smoke test — correct shapes/dtypes,
  bit-identical on repeated calls with the same seed (including the truth
  dict). Full production-scale pair (1000px reference + 1000px search,
  default supersample) completed in ~64s wall time in this dev environment
  — consistent with step 6's per-render timing observation, not a new
  slowdown. Rendered `docs/validation/step8_pair_sanity.png`: reference,
  full search image with the true reference location boxed, and a crop of
  the search image at `gt_search_px` — the crop visibly shows the same
  fin/gate grid structure as the reference, just coarser and noisier, which
  is the expected qualitative relationship and a strong end-to-end sanity
  check ahead of the formal gate tests.
- Note for step 9: confirmed the RNG design supports test 11's "same
  geometry, different noise seed" case, but *not* through
  `generate_pair`'s single top-level seed (changing `cfg.seed` changes
  geometry too). Test 11 will need to compose `build_geometry` /
  `optics.render` / `detector.apply_detector` directly rather than going
  through `generate_pair` twice with different seeds.
- Next: `validate.py` — the gate (tests 1, 6, 7). Need `scikit-image`
  (`skimage.registration.phase_cross_correlation`), not yet installed.

### 2026-08-16 — Step 9: validate.py, the gate (tests 1, 6, 7) — FAILING, stopped per protocol
- Installed `scikit-image` (also pulled in `imageio`, useful later for
  `make_dataset.py`).
- Implemented all three gate tests as pytest tests in `generator/validate.py`,
  writing `docs/validation/results.json` and (test 7)
  `docs/validation/test7_subpixel_linearity.png`. All "noiseless" renders use
  `dose_e_per_px=1e12` ("dose -> infinity" per L0, not a separate no-noise
  code path) and `quantise_8bit=False` for full-precision comparison.
- **Ran the suite: all three tests fail.**
  - Test 1 (physical correspondence): mean abs diff = **2.178 DN**, just
    over the 2.0 DN threshold.
  - Test 6 (2 nm / 0.2 search-px shift recovery): recovered **0.130 px**,
    outside 0.200 +/- 0.01.
  - Test 7 (linearity sweep): slope **1.125** (outside 1.00 +/- 0.02);
    residual peak-to-peak **0.127 px** against a 0.05 px step size — a
    clear staircase, not noise.
- Per CLAUDE.md: "If test 7 staircases even at k=16, stop and report." Did
  not tune the threshold to force a pass (explicit working-style rule).
  Instead ran a focused diagnostic before stopping, to have something
  concrete to report rather than just "it fails":
  - **Not a supersample/discretization issue.** Swept `k` at the search
    scale (10 nm/px, sigma=3 nm) from 8 to 32; max abs error across a 5-point
    shift sweep was **identical to 4 decimal places** (0.0800) at every k.
    Re-ran the same sweep at the *reference* scale (1 nm/px, sigma=1 nm)
    from k=4 to k=32: error was **exactly 0.6300 at every k** — completely
    invariant to supersample. This rules out "fine grid too coarse" as the
    cause.
  - **Not an image-size/boundary artifact.** 100 px vs 400 px search images
    gave essentially the same max error (0.08 vs 0.09).
  - **Somewhat sensitive to `psf_sigma_nm`, but no clean fix found.**
    sigma=3: 0.08, sigma=4: 0.06, sigma=5: 0.06, sigma=6: 0.07, sigma=8:
    0.14 (worse). There's a shallow, non-monotonic minimum around
    sigma=4-5 nm, not a threshold beyond which the problem disappears —
    contradicts a simple "PSF too narrow, widen it" fix.
  - **Present on an isolated single edge too, not just the periodic
    lattice** (tested with `fin_pitch_nm=2000` so only one edge is ever in
    view, same technique as step 6's isolated-edge test) — actually *worse*
    there (max error 0.25 vs 0.08-0.09 for the periodic lattice) and with a
    qualitatively different shape (recovered shift at true=0.5 was 0.25,
    not exact, unlike the periodic case where 0.0/0.5/1.0 all recovered
    exactly).
  - Cross-checked the measurement tool itself: `phase_cross_correlation`
    correctly recovers an exact FFT (`scipy.ndimage.fourier_shift`)
    sub-pixel shift of a synthetic periodic image to within ~0.01 px (after
    accounting for its documented shift-sign convention, which is the
    negative of the applied shift — `abs()` in the test code already
    handles that). So the tool itself is trustworthy on well-behaved input;
    the discrepancy is specific to images produced by this generator's
    render -> detector pipeline.
- Working hypothesis (not confirmed): this resembles "pixel-locking" /
  sub-pixel registration bias, a known real phenomenon when a pixelated
  imaging system's frequency response doesn't adequately suppress content
  near the Nyquist frequency before box integration — but the non-monotonic
  response to `psf_sigma_nm` and the fact it's *worse*, not better, on an
  isolated single edge don't fully fit that story either, and haven't been
  reconciled. Have **not** ruled out a genuine implementation bug (e.g. in
  how the sub-pixel origin shift interacts with `render()`'s pixel-center
  convention, or with `detector.py`'s Poisson step even at very high dose).
- **Stopping here per CLAUDE.md's explicit instruction, to report to the
  user rather than continue past a failing gate or self-adjust the
  threshold/defaults.** Steps 10-12 are on hold until this is resolved.

### 2026-08-16 — Step 9 follow-up: coordinate-system / subpixel-rendering bug audit
- User requested a structured audit to separate "implementation bug" from
  "genuine forward-model property" for the gate failures, explicitly
  forbidding threshold/PSF/physics tuning as part of this pass. All work
  below is read-only diagnostics in the scratchpad directory; **no file in
  `generator/` or `validate.py` was modified.**
- **Coordinate system audit (items 1-3, 8-10):** traced `render()`'s exact
  equations by hand: fine sample `j` sits at world
  `x0_nm + (j+0.5)*fine_step_nm` (midpoint rule); box-averaging `k`
  consecutive fine samples for output pixel `p` reduces exactly to the
  documented `x0_nm + (p+0.5)*pixel_size_nm` center convention — internally
  self-consistent, no half-pixel or boundary-convention bug found.
  `pair.py`'s `gt_search_px = (ref_center_nm - search_origin_nm) /
  pixel_size_nm` matches this convention, but is **not implicated** in the
  test 6/7 failure since those tests bypass `pair.py`/`gt_search_px`
  entirely and compare rendered images directly.
- **`subpixel_offset_px` semantics (item 6):** re-verified against D-019 —
  shifting both `ref_center_nm` and `search_origin_nm` together is correct
  by design (the one reading consistent with CLAUDE.md's own worked 0.200
  px example) and, again, not used by tests 6/7 at all (they shift
  `origin_nm` directly).
- **Decisive experiment (items 4-5, 7): isolated single edge, no LER/cuts/
  landmarks/periodicity, shifted at four different pipeline stages.**
  Built a minimal SDF (`d(x,y) = x`, exact distance to the line x=0),
  pushed it through the real `se_yield` -> PSF -> box-average chain, and
  compared four ways of producing a "shifted by t px" image:
  - **A** — re-evaluate the continuous SDF at shifted world coordinates and
    rerun the full pipeline (what `render()` actually does for a moved
    origin).
  - **B** — Fourier-shift the discretized *pre-PSF* fine yield field, then
    blur + box-average.
  - **C** — blur first, then Fourier-shift the *post-PSF* fine field, then
    box-average.
  - **D** — Fourier-shift the *finished, already box-averaged* image
    directly (bypasses the generator entirely; the "if there were zero
    generator-side bias" control).
  Result (search config: 10 nm/px, psf_sigma=3 nm, k=8):
  **D recovers every shift exactly (slope 1.0000, residual_ptp 0.0000)** —
  this holds on both the isolated edge and the real intel14 lattice, which
  conclusively rules out `phase_cross_correlation` bias and rules out
  periodicity/aliasing-of-the-lattice as the cause (an isolated edge has no
  periodicity and D is still exact). **A, B, and C all show the same
  S-curve bias as the original test 6/7 failure** (isolated edge: slope
  1.14-1.19, residual_ptp ~0.31; real lattice: slope 1.19, residual_ptp
  0.092, vs. the originally-observed slope 1.125/residual 0.127 on the full
  21-point sweep — consistent). B and C agree with each other exactly (as
  expected: Fourier shift commutes with a linear blur), which localizes the
  discrepancy to "decimating this content is not shift-equivariant," not to
  *when* in the pipeline the shift is applied.
- **k-invariance re-confirmed on the isolated edge specifically:** A's bias
  is identical (to ~0.01 px) at k=8, 16, 32 — supersample is not the
  limiting factor for tests 6/7. This is the discriminator: since the
  discrete integration has already converged by k=8, the remaining error
  cannot be quadrature/discretization error — it is a property of the
  *converged, continuous-domain* PSF -> box-average operation itself.
- **Separately, ran a k-sweep for test 1's own configuration** (matched
  psf_sigma_nm=1.0 at both scales — a different, more extreme regime than
  search's production sigma=3.0) and got the opposite signature: mean abs
  diff shrinks **monotonically and steeply** with the coarse capture's own
  supersample (k=8: 2.178 DN [matches the observed failure exactly], k=16:
  0.791, k=32: 0.265, k=64: 0.097 — comfortably under the 2.0 DN threshold
  by k=32). This is a **quadrature under-resolution artifact**, not the
  same phenomenon as tests 6/7: at psf_sigma_nm=1.0 nm with the coarse
  capture's fine step of 1.25 nm (k=8), `sigma_fine_px = 0.8` — the PSF's
  own standard deviation is *smaller than one fine-pixel spacing*, which is
  a badly under-resolved discrete approximation of a continuous Gaussian
  blur. (Search's *production* default, sigma=3.0 nm at the same pixel
  size, gives `sigma_fine_px = 2.4` at k=8 — already well converged, which
  is why tests 6/7 don't respond to k at all.)
- **Conclusion — two separate root causes, not one:**
  - **Test 1 is a genuine, fixable numerical bug** (under-resolved PSF
    quadrature at the specific matched-PSF/coarse-pixel combination this
    test uses). Raising supersample is the correct, spec-sanctioned fix
    (CLAUDE.md: "these must be configurable"), not a threshold hack.
  - **Tests 6/7 are a genuine forward-model property, not a bug.** The
    `se_yield` edge response has a real, deliberate jump discontinuity at
    d=0 (D-012 — intentional, meant to be smoothed only by the beam PSF).
    At the search capture's psf_sigma=3 nm / pixel=10 nm, that discontinuity
    survives Gaussian blurring with real spectral content above the pixel's
    Nyquist frequency; box-averaging (decimating) to the pixel grid is where
    the aliasing happens, and no amount of supersampling the pre-decimation
    integral removes it, because the bias lives in the converged continuous
    signal itself, not in how precisely we integrate it. This is exactly
    the contingency CLAUDE.md names for a test-7 staircase: "the PSF is too
    narrow relative to the sampling and we need to reconsider σ_beam for
    the search capture." (The earlier `psf_sigma_nm` sweep's non-monotonic
    result — improves 3->5 nm, worsens again by 8 nm — means simply
    widening sigma is not a clean drop-in fix on its own; it likely trades
    aliasing bias for the FOV-boundary reflect-mode bias noted in D-014,
    and needs to be re-examined together with FOV margin before picking a
    new value.)
- **No fix implemented.** Per the user's explicit instruction, this pass
  was diagnosis only — no threshold, PSF, LER, or physics parameter was
  changed. Decision on how to proceed (reconsider search σ_beam, and
  separately raise test 1's coarse-capture supersample) is pending user
  input. Steps 10-12 remain on hold.

### 2026-08-16 — Step 9, part 2: apply test 1's fix; controlled search-PSF sweep
- User authorized two concrete actions based on the audit above: (1) fix
  test 1 by raising its coarse-capture supersample (a numerics fix, no
  physics/threshold change), (2) run a *bounded* sigma sweep (3/5/8 nm,
  explicitly not open-ended, explicitly not "pick whichever passes") for
  tests 6/7's search PSF, with a boundary-vs-aliasing control, then decide
  and proceed — production `psf_sigma_nm` was explicitly *not* to be
  touched before this sweep ran.
- **Test 1 fix (D-020):** raised `test_1_physical_correspondence`'s
  `coarse_cfg` supersample 8 -> 32. Nothing else changed. Reran:
  **passes, 0.2645 DN < 2.0 DN threshold** (matches the k-sweep's k=32
  prediction of 0.265 DN exactly).
- **Controlled PSF sweep (D-021):** isolated single edge (no LER/cuts/
  periodicity) at sigma in {3,5,8} nm, each measured at a small FOV
  (2000 nm, edge centered, 1000 nm margin) *and* a 6x-larger FOV (12000 nm,
  6000 nm margin) to probe for `optics.py`'s documented reflect-mode
  boundary bias (D-014); plus the real intel14 lattice at the same three
  sigmas with a generous 500 nm extent margin (vs. the usual 100 nm); plus
  a dedicated sigma-fixed-at-3nm, FOV-only sweep (2000/4000/12000 nm).
  Results:
  | sigma | edge slope (small/large FOV) | edge residual ptp (small/large) | lattice slope | lattice residual ptp |
  |---|---|---|---|---|
  | 3 nm | 1.1299 / 1.1299 | 0.4605 / 0.4605 | 1.1013 | 0.1170 |
  | 5 nm | 1.0917 / 1.0917 | 0.3462 / 0.3462 | 1.0560 | 0.0640 |
  | 8 nm | 1.0548 / 1.0548 | 0.2518 / 0.2518 | 0.9680 | 0.0480 |
  - **Zero measurable boundary effect at any tested sigma** — small vs.
    large FOV results are identical to 4 decimal places at every sigma, and
    the dedicated FOV-only control (sigma fixed at 3, FOV 2000 -> 12000 nm)
    also showed zero change. This means the earlier (pre-this-audit)
    finding that sigma=8 was *worse* than sigma=3-5 was **not reproduced**
    under a properly boundary-controlled setup — here the trend is cleanly
    monotonically improving from 3 -> 5 -> 8 nm on both the isolated edge
    and the real lattice. The earlier non-monotonic result is judged to
    have been an artifact of a less-margined test setup at the time, not a
    real effect (not independently re-diagnosed further, per the user's
    explicit "do not spend more time on theoretical diagnosis").
  - Trend had **not plateaued by 8 nm** on either geometry — residual was
    still improving substantially from 5 -> 8 nm (e.g. lattice ptp
    0.064 -> 0.048, ~25% further reduction), so 8 nm is not obviously a
    point of "stability," just the best of the three authorized values.
  - **Decision:** adopt sigma=8 nm as the new search PSF (D-021) — best on
    every metric, no boundary-artifact downside found, consistent with
    CLAUDE.md's own "reconsider σ_beam" contingency. Updated
    `generator/validate.py::_SEARCH_PSF_SIGMA_NM` (was inline `3.0` in
    tests 6 and 7) and reran the *actual* tests (full 21-point sweep for
    test 7, not the quick approximation above).
  - **Actual rerun result at sigma=8 nm:**
    - **Test 6:** recovered 0.150 px (was 0.130), expected 0.200 +/- 0.01
      — **still fails**, gap shrank from 0.070 to 0.050 px.
    - **Test 7:** slope 1.0151 (was 1.1247) — **now passes** the
      1.00 +/- 0.02 criterion. Residual peak-to-peak 0.0868 px (was
      0.1274) — **still fails** the < 0.025 px criterion, though nearly
      halved.
  - **Tests 6 and 7 both still fail overall**, reported honestly rather
    than adjusted to pass. The improvement is real and substantial (slope
    error dropped ~8x; residual and test-6 gap both roughly halved), but
    incomplete. Since the trend had not plateaued at 8 nm and the sweep was
    explicitly bounded to {3,5,8} nm by the user, whether to extend the
    search further, accept this as a documented residual limitation, or
    invoke CLAUDE.md's option "C" (revise the validation criterion for
    this physical regime) is an open decision, not resolved here.
- **No threshold was modified. No empirical correction factor or
  calibration constant was introduced. The SE-yield discontinuity, ground-
  truth convention, and subpixel-shift implementation are all untouched.**
  Full numeric detail in `docs/DECISIONS.md` D-020/D-021 and
  `docs/validation/results.json`. Steps 10-12 remain on hold pending a
  decision on tests 6/7.

### 2026-08-16 — Step 9, part 3: production supersample set to k=16 (D-022)
- User noted test 1 already passes at k=16 (0.791 DN < 2.0, per D-020's own
  sweep) and asked for a single named "production coarse/search
  supersample" constant at k=16, with k=32/k=64 kept only as
  convergence-validation references, not values any test actually runs at.
- Added `validate.py::_PRODUCTION_SEARCH_SUPERSAMPLE = 16`; used it for
  test 1's `coarse_cfg` (was 32) *and* tests 6/7's `search_cfg` (was 8) for
  a single consistent value across the suite.
- Reran all three gate tests:
  - **Test 1: passes**, 0.7906 DN < 2.0 DN (matches the D-020 k=16
    prediction almost exactly).
  - **Test 6: still fails** — 0.150 px vs. 0.200 +/- 0.01, essentially
    identical to the k=8 result, confirming (again) the k-invariance
    established in the Step 9 audit.
  - **Test 7: still fails on residual** — slope 1.0122 (passes, was 1.0151
    at k=8), residual peak-to-peak 0.0755 px (was 0.0868 at k=8) against
    the 0.025 px threshold. The small k=8->k=16 shift is re-discretization
    noise, not a trend — consistent with, not contradicting, k-invariance.
- No threshold/PSF/geometry/detector parameter touched in this step —
  purely a render-cost decision now that k=16 is confirmed on the
  converged part of the quadrature curve. Full reasoning in D-022.
- Tests 6/7 remain open, exactly as reported after D-021: substantially
  improved from the original sigma=3/k=8 state but not passing. Decision
  on next steps (extend the sigma search, accept as a documented
  limitation, or revisit the validation criterion) still pending.

### 2026-08-17 — Steps 10-12: presets, make_dataset, remaining validation tests
- User instruction: proceed to Steps 10-12 keeping sigma_search=8nm and
  k=16 as-is; do not touch tests 6/7 further; treat the residual as a
  documented Phase 1 limitation; complete presets + dataset generation;
  stop once L0-L6 datasets are generated and sealed.
- `presets.py` written: `_geometry`/`_reference`/`_search` baseline
  builders, `L0`..`L6` functions implementing CLAUDE.md's ladder table
  (search PSF baseline revised per D-021, L4's sigma-delta translated
  onto that baseline, `edge_amp_scale` 0.9/1.1 split documented as our
  choice), `cut_density_case()` for the 5-point sweep, `LEVEL_PRESETS`
  dict. See D-023.
- `suites/make_dataset.py` written: writes `data/case_NNNN/{reference.png,
  search.png,config.json}` + `truth/case_NNNN/truth.json`; `config.json`
  deliberately omits `seed` as well as `ref_center_nm`/`search_origin_nm`
  (generate_pair is fully deterministic in (config, seed), so the seed
  alone would let someone reconstruct the truth by re-running the
  generator).
- **Blocker**: timing a production-scale (1000px) render at k=16 hit
  `numpy._core._exceptions._ArrayMemoryError` (tried to allocate 1.91 GiB
  for a single (16000,16000) array) -- the fine grid at full scale is far
  bigger than the small validation-gate renders that motivated k=16.
  Presented three options to the user; chose k=8 for shipped data,
  keeping k=16 only for the small-scale gate tests. See D-024 for the
  full numeric justification that k=8 stays on the converged/safe side of
  the quadrature curve for every shipped preset's actual
  psf_sigma_nm/pixel_size_nm ratio.
- Dataset generation batched (2 cases per background invocation, ~185s/
  case at k=8/1000px) to stay under per-invocation time limits; 33 cases
  total (28 across L0-L6 at 4/level + 5 for the cut_density sweep at
  {0.05,0.02,0.01,0.005,0.0}).
- Wrote the remaining validation tests (2,3,4,5,8,9,10,11). First combined
  run of 3/4/5/8/9 hung indefinitely -- diagnosed as tests 3/4's isolated-
  edge fixture using out_px=400 over a 40nm FOV (0.1nm/px) with k=16,
  which drives `sigma_fine_px` up to 640 and makes
  `scipy.ndimage.gaussian_filter`'s cost (which scales with kernel
  radius) explode; confirmed via `Get-Process` CPU accumulation that it
  was genuinely computing, not deadlocked. Fixed by dropping to k=4 (no
  accuracy cost at this pixel/sigma ratio) -- runtime dropped to ~150s.
- That fix then exposed two real, previously-masked bugs in the fixture
  itself: (1) the real intel14 fin (8nm wide) put the edge's own opposite
  edge inside the PSF's influence radius at large sigma, breaking
  isolation; (2) even after widening the fixture geometry, test 3's
  width metric (20-80% of profile.min()/max()) was wrong for a yield
  model where `Y_SUBSTRATE==Y_FEATURE` -- profile.min()/max() are the
  edge-enhancement peak/trough, whose *amplitude* shrinks with PSF blur
  even as their true *separation* grows, making the naive metric
  non-monotonic. Switched to peak-to-trough x-separation directly
  (verified monotonic: 2.10/3.70/6.70/12.50 nm at sigma=0.5/1.0/2.0/4.0).
  Full writeup in D-025.
- Added a `preset="custom"` escape hatch to `config.resolve_preset()`
  (bypasses the NODE_PRESETS lookup, returns the geometry unchanged) so
  the widened isolated-edge fixture doesn't misrepresent itself as a real
  process node in the public preset table.
- Final full-suite run (`pytest generator/validate.py`, all 11 tests):
  **9 passed, 2 failed (tests 6/7, exactly as documented in D-021/D-022 --
  0.150px vs 0.200+/-0.01, residual_ptp 0.0755px)**. No threshold, PSF, or
  forward-model parameter was changed to reach this state; the two
  failures are the accepted, documented Phase 1 limitation per explicit
  user instruction. `docs/validation/results.json` now has all 11 test
  entries.
- Dataset generation resumed in batches; proceeding toward all 33 cases.

### 2026-08-17 — Dataset generation complete, Phase 1 sealed
- All 33 cases generated: `data/case_0000`..`case_0032`,
  `truth/case_0000`..`truth/case_0032` (28 across L0-L6 at 4/level, 5 for
  the cut_density sweep at {0.05,0.02,0.01,0.005,0.0}).
- Verified programmatically across all 33 cases: every `config.json` has
  exactly `{level, geometry, reference, search}` keys with no `seed`,
  `ref_center_nm`, or `search_origin_nm` at any nesting depth (checked
  recursively, not just top-level); every `truth.json` has exactly the
  required keys (`ref_center_nm`, `search_origin_nm`, `gt_search_px`,
  `lattice_phase_fin`, `lattice_phase_gate`,
  `dist_to_nearest_landmark_nm`); every case has all four files
  (reference.png, search.png, config.json, truth.json).
- Generated `docs/validation/contact_sheet_L0_L2_L6.png` from case_0000
  (L0), case_0008 (L2), case_0024 (L6) -- reference+search side by side
  per level. Visual sanity: L0 reference shows the clean fin/gate grid
  with landmarks; L2 shows added dose noise and a visible large cut
  feature; L6 shows a perfectly uniform lattice (no landmarks/cuts, as
  specified) and clearly visible row banding in its search image
  (row_band_amp=4.0). All physically consistent with the intended
  difficulty ladder.
- Regenerated `docs/assumptions.md` (`grep -rn "# ASSUMPTION:" generator/`)
  -- picked up `presets.py`'s L4 sigma-delta assumption, which the
  previous copy predated.
- **Phase 1 status**: 9/11 validation tests pass; tests 6/7 fail exactly
  as documented in D-021/D-022/D-025 (recovered 0.150px vs 0.200+/-0.01;
  residual_ptp 0.0755px vs <0.025px), an explicitly accepted, documented
  limitation per direct user instruction, not an oversight. All other
  Definition-of-Done items (sealed dataset, determinism, config/truth
  separation, contact sheet, DECISIONS/LOG/sources/assumptions docs) are
  met. Stopping here per the user's explicit instruction -- no Phase 2
  (localization) work started.
