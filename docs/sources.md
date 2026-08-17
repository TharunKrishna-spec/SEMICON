# Sources

Every literature-derived constant used in the generator, with provenance and confidence.

- **(a) sourced** — a specific paper/patent/disclosure states this value
- **(b) inferred** — reasonable derivation from a sourced value
- **(c) our choice** — engineering judgment, not from literature

## Node geometry presets

| Constant | Value | Tag | Source |
|---|---|---|---|
| `intel14` fin pitch | 42 nm | (a) | Intel 14 nm published figures |
| `intel14` fin width | 8 nm | (a) | Intel 14 nm published figures |
| `intel14` fin height | 42 nm | (a) | Intel 14 nm published figures |
| `intel14` gate pitch | 70 nm | (a) | Intel 14 nm published figures |
| `intel14` gate length | 20 nm | (a) | Intel 14 nm published figures |
| `n7` fin pitch | 30 nm | (a) | IBM/imec disclosures |
| `n7` fin width | 6 nm | (a) | IBM/imec disclosures |
| `n7` fin height | 45 nm | (a) | IBM/imec disclosures |
| `n7` gate pitch (CPP) | 48 nm | (a) | IBM/imec disclosures (CPP 44/48 nm range) |
| `n7` gate length | 18 nm | (a) | IBM/imec disclosures |
| `n5` fin pitch | 28 nm | (a) | IBM/imec disclosures |
| `n5` fin width | 6 nm | (a) | IBM/imec disclosures |
| `n5` fin height | 50 nm | (a) | IBM/imec disclosures |
| `n5` gate pitch | 44 nm | (a) | IBM/imec disclosures |
| `n5` gate length | 16 nm | (a) | IBM/imec disclosures |

## SE yield / edge response (Mack-Bunday ALM, Si-on-Si, 500 eV)

| Constant | Value | Tag | Source |
|---|---|---|---|
| `Y_SUBSTRATE` | 0.817 | (a) | Mack-Bunday analytical linescan model, Si-on-Si, 500 eV |
| `Y_FEATURE` | 0.817 | (a) | Mack-Bunday analytical linescan model, Si-on-Si, 500 eV |
| `ALPHA_E` | 1.65 | (a) | Mack-Bunday ALM edge enhancement factor |
| `SIGMA_E` | 2.66 nm | (a) | Mack-Bunday ALM step forward-scatter range |
| `ALPHA_V` | 0.64 | (a) | Mack-Bunday ALM volume loss factor |
| `SIGMA_V` | 0.26 nm | (a) | Mack-Bunday ALM volume loss range |
| `PHI_F` | 0.245 | (a) | Mack-Bunday ALM substrate forward-scatter absorption |
| `LAMBDA_F` | 1.95 nm | (a) | Mack-Bunday ALM wafer forward-scatter range |
| `PHI_B` | 0.22 | (a) | Mack-Bunday ALM substrate backscatter absorption |
| `LAMBDA_B` | 0.82 x fin_height_nm | (a) | Mack-Bunday ALM backscatter range per step height |

Applying a Si-on-Si, 90 deg sidewall, 20-100 nm feature height fit to a generic
FinFET stack (Si fin on STI oxide, HKMG gate) is an approximation we have not
calibrated against real tool data. See comment in `generator/yield_field.py`.

## Beam PSF

| Constant | Value | Tag | Source |
|---|---|---|---|
| CD-SEM probe size | 1-5 nm | (a) | Typical CD-SEM specifications |
| Landing energy | 300-800 eV | (a) | Typical CD-SEM specifications |
| FWHM -> sigma range | 0.85-1.7 nm | (b) | Derived from FWHM 2-4 nm via sigma = FWHM / 2.355 |
| Reference `psf_sigma_nm` | 1.0 nm | (b) | Mid-range of derived sigma, higher-magnification capture assumed better focused |
| Search `psf_sigma_nm` | 8.0 nm | (c) | Revised from CLAUDE.md's literal 3.0 nm default after a boundary-controlled sigma sweep (3/5/8 nm) showed 8 nm was the best-performing, no-boundary-artifact option among those tested for tests 6/7's subpixel-recovery gate; still a further extension beyond the literal 0.85-1.7 nm probe-derived range than 3.0 nm was, on the same "lower magnification, less well focused" physical basis. Does not by itself clear the test 6/7 thresholds -- see D-021. |

## Line-edge roughness

| Constant | Value | Tag | Source |
|---|---|---|---|
| LER 3-sigma | 2-5 nm range, 3.0 nm default | (a)/(c) | Range from CD-SEM LER literature (a); default point choice (c) |
| LER correlation length | 25 nm | (c) | Engineering choice; Gaussian-correlated model selected over power-law PSD, see D-006/D-007 |
| LER sampling grid step | 2 nm | (c) | Our choice, resolves the default 25 nm correlation length at ~12 samples/length; see D-006 and the `# ASSUMPTION:` comment in `generator/geometry.py` |

## Supersampling

| Constant | Value | Tag | Source |
|---|---|---|---|
| Reference `k` | 4 (fine step 0.25 nm) | (c) | Our choice, validated by gate test 7 |
| Search `k` | 8 (fine step 1.25 nm) | (c) | Our choice, validated by gate test 7 |

## Cuts and landmarks

| Constant | Value | Tag | Source |
|---|---|---|---|
| Cut length fraction | 0.3 x perpendicular pitch | (c) | Our choice; see D-010 |
| Cut candidate-site placement | gap midpoints between perpendicular-feature crossings | (c) | Our choice, physically motivated by real fin-cut/gate-cut litho; see D-009 |
| Landmark shape | axis-aligned square, side = `landmark_scale_nm` | (c) | Our choice; see D-011 |
| Landmark placement | uniform random within 80% of `extent_nm` | (c) | Our choice; see D-011 |

## Detector

| Constant | Value | Tag | Source |
|---|---|---|---|
| `_BASE_DN_PER_YIELD` | 100 DN per unit yield | (c) | Our choice, uncalibrated -- CLAUDE.md explicitly excludes calibrated dose/grey-level; see D-015 |
| `_BAND_CORR_LEN_PX` | 15 output px | (c) | Our choice; see D-016 |

## Capture-asymmetry defaults (dose, read noise, banding, etc.)

All default numeric ranges for `CaptureConfig` fields not listed above
(dose, read-noise sigma, brightness offset, contrast gain, banding amplitude)
are **(c) our choice**, picked to be plausible acquisition-to-acquisition
variation, not calibrated to a specific tool.
