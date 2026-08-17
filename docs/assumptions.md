# Assumptions

Auto-generated from `# ASSUMPTION:` comments in `generator/`. Regenerate with:

```
grep -rn "# ASSUMPTION:" generator/ > docs/assumptions.md
```

(Header lines above this point are added back manually after regeneration.)

generator/config.py:12:# ASSUMPTION: node presets below are taken from published/disclosed figures
generator/detector.py:19:# ASSUMPTION: CLAUDE.md explicitly excludes "calibrated dose->grey-level"
generator/detector.py:28:# ASSUMPTION: row banding needs a correlation length along the row axis for
generator/geometry.py:45:# ASSUMPTION: the LER sampling grid step is fixed at 2 nm, chosen to resolve
generator/geometry.py:52:# ASSUMPTION: a cut's length along the feature it severs is set to 30% of
generator/geometry.py:59:# ASSUMPTION: landmarks are placed uniformly at random within 80% of
generator/geometry.py:324:# ASSUMPTION: exact only away from fin/gate crossings and cut/landmark
generator/pair.py:22:# ASSUMPTION: the search capture's field of view is anchored (centered) at
generator/yield_field.py:17:# ASSUMPTION: these are the Mack-Bunday analytical linescan model (ALM)
