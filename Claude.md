## Project

Build a physically defensible synthetic SEM image generator for a nanoscale search-and-localize benchmark (Applied Materials / SEMICON India 2026 student hackathon).

The generator produces **pairs** of grayscale images of the same physical FinFET die region:

- **Reference:** 1000×1000 px, 1 nm/px, covering 1 µm × 1 µm
- **Search:** 1000×1000 px, 10 nm/px, covering 10 µm × 10 µm, containing the reference region

A separate Phase 2 (not in scope here) will localize the reference inside the search image and return the center (x, y).

**Your job is the generator and its validation suite. Do not write any localization or matching code.** If you find yourself writing `matchTemplate`, `phase_cross_correlation` for anything other than validation tests 6 and 7, or anything resembling a matcher, stop — that is Phase 2.

## Why this matters (read this, it changes design decisions)

This generator is a **measuring instrument**. Phase 2 will use it to compare intensity-NCC, gradient-NCC, phase correlation, and possibly learned matchers on identical image pairs. If the generator has a sampling artifact, the benchmark silently measures the artifact instead of the algorithms — and still produces plausible-looking numbers. Correctness of the *forward model ordering* matters more than visual realism.

Concretely: do not add an effect because it looks more like a real SEM. Add it only if it changes the information a localization algorithm can extract.

---

## Non-negotiable constraints

1. **Geometry lives in physical nanometres**, as analytic primitives. Never as a raster. The only query the renderer makes is *signed distance to nearest edge* at arbitrary continuous (x, y).

2. **Never resize a finished image to make the other one.** Both captures render independently from the same continuous geometry model, each with its own PSF, gain, and noise.
   *Clarification, because this is easy to get wrong:* supersampling internally and box-averaging k×k blocks **is** correct — that is exact pixel-area integration, not resizing. What is forbidden is taking the finished, noisy reference and interpolating it down.

3. **Forward model order is fixed:**
   ```
   geometry → SE yield field (incl. edge response) → PSF → pixel-area
   integration → gain/offset → Poisson → Gaussian read → banding →
   clip → 8-bit quantize
   ```
   PSF before integration. Noise after gain. Getting this order wrong is the single most likely bug and it will not be visually obvious.

4. **Shared vs independent is fundamental.**
   - Shared (same physical die): geometry, LER realisation, lattice phase, landmarks, true reference center.
   - Independent (separate acquisitions): noise seed, dose, read-noise σ, brightness offset, contrast gain, PSF σ, edge-response amplitude, banding realisation.
   - Enforce this **structurally in the API** — it must be impossible to pass different geometry to the two captures.

5. **Ground truth is defined in nanometres first**, pixels derived. It must be continuous (non-integer) by default, and written to a directory the generation config does not reference.

6. **Determinism.** One seed → bit-identical output. Use `np.random.default_rng(seed).spawn(n)` for independent-but-reproducible streams.

---

## Module layout

```
generator/
    __init__.py
    config.py        # dataclasses
    geometry.py      # primitives, node presets, signed distance, LER, cuts
    yield_field.py   # SE yield from signed distance (ALM two-exponential)
    optics.py        # Gaussian PSF + supersample/area-integrate
    detector.py      # gain, offset, Poisson, Gaussian, banding, quantize
    pair.py          # generate_pair(cfg) -> (ref, search, truth)
    presets.py       # intel14/n7/n5 geometry; L0..L6 ladder configs
    validate.py      # the validation tests
suites/
    make_dataset.py  # writes data/ and truth/ from a config list
docs/
    DECISIONS.md     # append-only decision record  (see below)
    LOG.md           # append-only build log
    assumptions.md   # auto-generated from `# ASSUMPTION:` comments
    sources.md       # every literature-sourced constant + where it came from
    validation/      # test output: plots, contact sheets, numeric results
data/                # images + config.json  (no ground truth)
truth/               # truth.json only
```

Dependencies: numpy, scipy (`gaussian_filter`, `gaussian_filter1d`), imageio or opencv-python, pytest. Nothing else — `pip freeze` goes in the submission.

---

## Physics constants — use these, do not invent

### Node geometry presets

| Preset | fin pitch | fin width | fin height | gate pitch | gate length |
|---|---|---|---|---|---|
| `intel14` | 42 nm | 8 nm | 42 nm | 70 nm | 20 nm |
| `n7` | 30 nm | 6 nm | 45 nm | 48 nm | 18 nm |
| `n5` | 28 nm | 6 nm | 50 nm | 44 nm | 16 nm |

Source: Intel 14 nm published figures; 7 nm CPP 44/48 nm and metal pitch 36 nm from IBM/imec disclosures. Default to `intel14`.

### SE yield / edge response (Mack–Bunday analytical linescan model, Si-on-Si, 500 eV)

```python
Y_SUBSTRATE = 0.817
Y_FEATURE   = 0.817
ALPHA_E     = 1.65    # edge enhancement factor
SIGMA_E     = 2.66    # nm, step forward-scatter range
ALPHA_V     = 0.64    # volume loss factor
SIGMA_V     = 0.26    # nm, volume loss range
PHI_F       = 0.245   # substrate forward-scatter absorption
LAMBDA_F    = 1.95    # nm, wafer forward-scatter range
PHI_B       = 0.22    # substrate backscatter absorption
LAMBDA_B    = 0.82 * fin_height_nm   # backscatter range per step height
```

Yield model, with `d` = signed distance (positive inside feature):

```
inside  (d > 0):  Y = Y_FEATURE  + ALPHA_E*exp(-d/SIGMA_E) - ALPHA_V*exp(-d/SIGMA_V)
outside (d ≤ 0):  Y = Y_SUBSTRATE - PHI_F*exp(d/LAMBDA_F)  - PHI_B*exp(d/LAMBDA_B)
```

Put a comment in `yield_field.py` noting these are fitted for silicon-on-silicon at 500 eV, 90° sidewall, 20–100 nm feature height, and that applying them to a generic FinFET stack is an approximation. We must not claim calibration we have not done.

### Beam PSF

CD-SEM probe size 1–5 nm; landing energy typically 300–800 eV. Use FWHM 2–4 nm → **σ_beam ≈ 0.85–1.7 nm**.

Defaults: reference `psf_sigma_nm = 1.0`, search `psf_sigma_nm = 3.0` (lower magnification is typically less well focused — state as an assumption).

### Line-edge roughness

3σ of 2–5 nm; default **3.0 nm**, correlation length 25 nm. Gaussian-correlated is sufficient; power-law PSD is a bonus.

### Supersampling

Reference `k = 4` (fine step 0.25 nm). Search `k = 8` (fine step 1.25 nm). These must be configurable — validation test 7 may force them higher.

---

## Config dataclasses

```python
@dataclass
class GeometryConfig:
    preset: str = "intel14"
    fin_pitch_nm: float = 42.0
    fin_width_nm: float = 8.0
    fin_height_nm: float = 42.0
    gate_pitch_nm: float = 70.0
    gate_length_nm: float = 20.0
    rotation_deg: float = 0.0        # keep small, 0-2
    corner_radius_nm: float = 2.0
    ler_sigma3_nm: float = 3.0
    ler_corr_len_nm: float = 25.0
    cut_density: float = 0.02        # 0 => intrinsically ambiguous
    landmark_count: int = 2
    landmark_scale_nm: float = 150.0

@dataclass
class CaptureConfig:
    pixel_size_nm: float             # 1.0 or 10.0
    size_px: int = 1000
    supersample: int = 8
    psf_sigma_nm: float = 1.0
    dose_e_per_px: float = 800.0
    read_noise_sigma: float = 2.0    # DN
    brightness_offset: float = 0.0   # DN
    contrast_gain: float = 1.0
    edge_amp_scale: float = 1.0
    row_band_amp: float = 0.0        # DN, 0 disables
    quantise_8bit: bool = True

@dataclass
class PairConfig:
    geometry: GeometryConfig
    reference: CaptureConfig
    search: CaptureConfig
    ref_center_nm: tuple | None = None    # None => random placement
    subpixel_offset_px: tuple = (0.0, 0.0)
    seed: int = 0
```

`generate_pair(cfg) -> (ref_u8, search_u8, truth_dict)`

`truth_dict` must contain at minimum: `ref_center_nm`, `search_origin_nm`, `gt_search_px` (float), plus diagnostics `lattice_phase_fin`, `lattice_phase_gate`, `dist_to_nearest_landmark_nm`.

---

## Build order — follow this exactly

Work in small commits. After each numbered step: print a one-line summary of what changed and what you verified, **and append to `docs/LOG.md` plus any `docs/DECISIONS.md` entries that step produced.**

0. Repo skeleton + `docs/` with empty `DECISIONS.md`, `LOG.md`, `sources.md`, and `validation/`. Seed `sources.md` with the constants given in this prompt, tagged (a)/(b)/(c).
1. `config.py` — dataclasses above, plus a `resolve_preset()` that fills geometry from the preset name.
2. `geometry.py` — fins + gates as rotated rectangles; `signed_distance(model, X, Y)` vectorised over NumPy arrays. Start with **no LER, no cuts**. Verify by rendering a raw distance field to PNG and eyeballing it.
3. `geometry.py` — LER. Generate once per model, in physical coordinates, stored on the model object. Perturb edge positions when computing distance.
4. `geometry.py` — cuts and landmarks, driven by `cut_density` and `landmark_count`.
5. `yield_field.py` — the two-exponential model above.
6. `optics.py` — `render(model, origin_nm, fov_nm, out_px, psf_sigma_nm, k)`. Supersample, apply `gaussian_filter` at the fine grid in fine-pixel units, then `reshape(out, k, out, k).mean(axis=(1,3))`.
7. `detector.py` — gain/offset → `rng.poisson` → `rng.normal` → optional banding → clip → uint8.
8. `pair.py` — RNG stream splitting, shared geometry construction, both renders, truth dict.
9. **`validate.py` — implement tests 1, 6, 7 and run them. Do not continue until they pass.** (See gate below.)
10. `suites/make_dataset.py` — writes `data/case_NNNN/{reference.png,search.png,config.json}` and `truth/case_NNNN/truth.json`. `config.json` must **not** contain `ref_center_nm` or `search_origin_nm`.
11. Remaining validation tests (2, 3, 4, 5, 8, 9, 10, 11).
12. `presets.py` — L0…L6 ladder configs.

---

## The gate — tests 1, 6, 7

**Nothing downstream is meaningful until these three pass.** Implement them as pytest tests.

**Test 1 — physical correspondence.** Render the reference region noiseless at 1 nm/px. Render the *same* 1 µm region noiseless at 10 nm/px. Area-average the reference 10×10. Mean absolute difference must be **< 2 DN**.

**Test 6 — subpixel shift recoverable.** Noiseless. Render a pair. Shift `ref_center_nm` by exactly 2 nm in x (= 0.2 search px) and re-render. Recover the shift with `skimage.registration.phase_cross_correlation(upsample_factor=100)`. Must return **0.200 ± 0.01 px**.
*(This is the one use of a correlation function permitted in Phase 1 — it is a measuring tool here, not a matcher.)*

**Test 7 — subpixel linearity.** Sweep the offset 0 → 1.0 px in 0.05 steps. Plot recovered vs true. Fit a line: slope must be **1.00 ± 0.02**, and the residuals must show **no staircase**. A staircase means aliasing — raise `supersample` and re-run.

If test 7 staircases even at `k = 16`, stop and report. That indicates the PSF is too narrow relative to the sampling and we need to reconsider σ_beam for the search capture.

## Remaining validation tests

2. Dose→SNR: sweep dose 50…5000, SNR in a flat region ∝ √dose within 5%.
3. PSF broadens edges: sweep σ_beam 0.5…4 nm, 20–80% edge width increases monotonically.
4. Edge response: toggle `ALPHA_E` 0 → 1.65, peak-to-plateau ratio rises from ~1.0 to ~1.5–1.8.
5. Banding is row-structured: row-mean autocorrelation shows the injected length, column-mean stays flat.
8. No moiré: `cut_density=0`, FFT of search shows fin fundamental + harmonics only, no low-frequency beat.
9. Shared LER: extract an edge-displacement profile for one fin from each capture; cross-correlation must be positive. Zero means LER was resampled independently — a bug.
10. Determinism: same seed twice → bit-identical.
11. Independence: same geometry, different seeds → identical geometry, noise correlation |r| < 0.05.

---

## Documentation and decision log

Create `docs/` at Step 1, before writing any generator code. It is not an afterthought — Phase 4 documentation is a graded deliverable, and reconstructing *why* we chose something three weeks later is far more expensive than writing one paragraph now.

### `docs/DECISIONS.md` — append-only

One entry every time you make a choice that a reasonable engineer could have made differently. Never edit or delete a past entry; if a decision is reversed, add a new entry that supersedes it and say so.

Format:

```markdown
## D-007 — Gaussian-correlated LER instead of power-law PSD
**Date:** 2026-08-16
**Status:** accepted
**Context:** LER needs a spatial correlation structure. CD-SEM roughness
literature characterises LER via a power-law PSD split into low/mid/high
frequency bands.
**Decision:** Use Gaussian-correlated roughness with a single correlation
length (default 25 nm).
**Alternatives considered:** Full power-law PSD; white-noise per-edge-point.
**Rationale:** Power-law is ~40 lines and needs its own validation; Gaussian
correlation captures the property Phase 2 actually depends on (each fin
period becomes distinguishable at 1 nm/px but not at 10 nm/px). White noise
would be washed out entirely by the PSF.
**Consequences:** Our LER spectrum is not spectrally realistic. If Phase 2
results turn out sensitive to roughness spectrum, revisit.
**Supersedes:** —
```

Log a decision for at least: the SDF formulation for rotated rectangles; how LER perturbs the distance field; supersampling factors; the search-capture PSF sigma; how cuts and landmarks are placed; the RNG stream-splitting scheme; what goes in `config.json` vs `truth.json`; any validation threshold you set.

Also log **rejections** — "we did not implement X because Y." Those are worth as much as the acceptances when writing Phase 4 and when defending scope to a judge.

### `docs/LOG.md` — append-only

Short, chronological, one block per work session or per completed step. This is a lab notebook, not prose.

```markdown
### 2026-08-16 — Step 2: geometry + SDF
- Implemented rotated-rectangle signed distance, vectorised.
- Rendered raw distance field → docs/validation/step2_sdf.png. Looks correct.
- Gotcha: rotation must be applied to the query point, not the rectangle,
  or the SDF is wrong for |theta| > 0. Cost ~30 min.
- Open: corner rounding not yet implemented, fins are sharp-cornered.
- Next: LER.
```

Record surprises and dead ends explicitly. "We tried A, it failed because B, so we did C" is the single most useful thing in the file later, and it is the thing everyone forgets to write down.

### `docs/sources.md`

Every literature-derived constant in the codebase, with where it came from and how confident we are. Mark each entry as one of:

- **(a) sourced** — a specific paper/patent/disclosure states this value
- **(b) inferred** — reasonable derivation from a sourced value
- **(c) our choice** — engineering judgment, not from literature

This maps directly onto the Phase 4 writeup and protects us if a judge challenges a number. The ALM constants and node geometries in this prompt are (a); the search-capture PSF sigma is (b); most capture-asymmetry ranges are (c).

### `docs/assumptions.md`

Generate from the `# ASSUMPTION:` comments in the source:

```bash
grep -rn "# ASSUMPTION:" generator/ > docs/assumptions.md
```

Add a make target or a one-line script. Regenerate it whenever you add assumptions.

### `docs/validation/`

Every validation test writes its artifacts here — the test-7 linearity plot, the test-8 FFT, the contact sheets, and a `results.json` with the numeric outcomes. Phase 4 needs these as figures; do not make them ephemeral matplotlib windows.

### Rule

**A step is not complete until its DECISIONS/LOG entries are written.** Treat them the way you treat the tests: part of the work, not documentation of the work.

---

## Difficulty ladder (`presets.py`)

| Level | Configuration |
|---|---|
| L0 | noiseless, matched gain, PSF on, dose→∞ (skip Poisson) |
| L1 | Poisson + Gaussian, symmetric (both dose 800) |
| L2 | asymmetric: ref dose 2000, search dose 300 |
| L3 | L2 + search `brightness_offset=15`, `contrast_gain=0.75` |
| L4 | L3 + ref `psf_sigma_nm=1.0`, search `psf_sigma_nm=3.5`, `edge_amp_scale` differing 0.9/1.1 |
| L5 | L4 + search `row_band_amp=4.0` |
| L6 | L5 + `cut_density=0.0`, `landmark_count=0`, reference forced near lattice center |

Also emit a `cut_density` sweep: {0.05, 0.02, 0.01, 0.005, 0.0} at otherwise-fixed L2 settings, for the Phase 3 failure-boundary characterisation.

---

## Do not implement

Monte Carlo electron transport (JMONSEL). Sample charging. Full ALM with footing and corner-rounding parameter fits. Backscatter channel. Detector take-off geometry. Calibrated dose→grey-level. GAN realism. Drift distortion. Frequency-domain MTF modelling. 3-D topographic rendering.

If you think one of these is needed, say so and explain what localization-relevant information it would add — do not just add it.

---

## Working style

- Small commits, one concern each.
- Vectorise with NumPy; no per-pixel Python loops in the renderer.
- Type hints on public functions. Docstrings state **units** (nm vs px vs DN) — most bugs in this codebase will be unit confusion.
- Write the validation test alongside the feature, not after.
- When you make a physics assumption not covered above, add a `# ASSUMPTION:` comment explaining it. These get collected into the Phase 4 writeup, where we must distinguish literature-sourced values from our own choices.
- If a test fails, do not tune the threshold to make it pass. Report it.

## Definition of done for Phase 1

- [ ] `generate_pair` produces a valid pair for all L0–L6 presets
- [ ] Validation tests 1, 6, 7 pass
- [ ] Remaining validation tests pass or have documented, understood failures
- [ ] `make_dataset.py` produces sealed `data/` + `truth/` for ≥30 cases
- [ ] `config.json` provably cannot reconstruct the answer
- [ ] Same seed reproduces bit-identically
- [ ] A contact sheet PNG of one L0, one L2, and one L6 pair for visual sanity
- [ ] `docs/DECISIONS.md` covers every non-obvious choice, including rejections
- [ ] `docs/LOG.md` is current through the final step
- [ ] `docs/sources.md` tags every constant (a)/(b)/(c)
- [ ] `docs/assumptions.md` regenerated
- [ ] `docs/validation/` holds plots and `results.json` for all tests

Then stop. Phase 2 begins in a fresh session.

---

## First action

Read this file, then do Step 0 (repo skeleton + `docs/`), Step 1, and Step 2. Show me the raw signed-distance field render before adding anything on top of it, and show me the first `docs/LOG.md` entry.