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
