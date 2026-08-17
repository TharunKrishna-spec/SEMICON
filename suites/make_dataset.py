"""Step 10: writes sealed `data/` + `truth/` from a list of generation
cases. `config.json` (in `data/case_NNNN/`) describes only the generation
*settings* -- geometry and capture configs, difficulty level label -- and
never the per-case `seed`, `ref_center_nm`, or `search_origin_nm`. Since
`generate_pair` is fully deterministic in `(config, seed)`, omitting
`ref_center_nm`/`search_origin_nm` alone would not be enough: anyone who
also had the seed could just re-run `generate_pair` and recover them. The
seed therefore lives only in this script's in-memory case list, never
written to `data/` or anywhere else on disk -- `truth/case_NNNN/truth.json`
is the only place `ref_center_nm`/`search_origin_nm` (and everything
derived from them) are recorded.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import imageio.v3 as iio

from generator import presets
from generator.config import PairConfig
from generator.pair import generate_pair

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRUTH_DIR = REPO_ROOT / "truth"


def _config_json(level: str, cfg: PairConfig) -> dict:
    """Everything about how this case was generated *except* the seed and
    the placement fields (`ref_center_nm`, `search_origin_nm`) -- see
    module docstring.
    """
    return {
        "level": level,
        "geometry": dataclasses.asdict(cfg.geometry),
        "reference": dataclasses.asdict(cfg.reference),
        "search": dataclasses.asdict(cfg.search),
    }


def write_case(case_id: str, level: str, cfg: PairConfig) -> None:
    """Render one (reference, search, truth) triple and write it to
    `data/<case_id>/` and `truth/<case_id>/truth.json`.
    """
    ref_image, search_image, truth = generate_pair(cfg)

    case_data_dir = DATA_DIR / case_id
    case_data_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(case_data_dir / "reference.png", ref_image)
    iio.imwrite(case_data_dir / "search.png", search_image)
    (case_data_dir / "config.json").write_text(json.dumps(_config_json(level, cfg), indent=2))

    case_truth_dir = TRUTH_DIR / case_id
    case_truth_dir.mkdir(parents=True, exist_ok=True)
    (case_truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))


def build_case_list(cases_per_level: int = 5, base_seed: int = 1000) -> list[tuple[str, str, PairConfig]]:
    """`(case_id, level_label, PairConfig)` for every L0-L6 case plus the
    cut_density sweep (CLAUDE.md: {0.05, 0.02, 0.01, 0.005, 0.0} at
    otherwise-fixed L2 settings), each case getting its own seed so
    geometry/LER/cuts/landmarks/noise/placement all vary across cases
    within a level (determinism is per-case, not "every L2 case is
    identical").
    """
    cases: list[tuple[str, str, PairConfig]] = []
    case_num = 0
    seed = base_seed

    for level, level_fn in presets.LEVEL_PRESETS.items():
        for _ in range(cases_per_level):
            case_id = f"case_{case_num:04d}"
            cases.append((case_id, level, level_fn(seed=seed)))
            case_num += 1
            seed += 1

    for cut_density in presets.CUT_DENSITY_SWEEP:
        case_id = f"case_{case_num:04d}"
        label = f"cutsweep_{cut_density:.3f}"
        cases.append((case_id, label, presets.cut_density_case(cut_density, seed=seed)))
        case_num += 1
        seed += 1

    return cases


def main(cases_per_level: int = 4) -> None:
    # 4 cases/level x 7 levels (28) + 5 cut_density-sweep cases = 33,
    # clearing the >=30-case Definition-of-Done floor with a small margin.
    # Kept deliberately close to that floor: each case costs ~185s of pure
    # render time at production scale (k=8 -- see D-024's memory-driven
    # supersample choice), so the full run is already ~100+ minutes.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_case_list(cases_per_level=cases_per_level)
    print(f"Generating {len(cases)} cases...")
    for i, (case_id, level, cfg) in enumerate(cases):
        write_case(case_id, level, cfg)
        print(f"  [{i + 1}/{len(cases)}] {case_id} ({level}) written")
    print(f"Done. {len(cases)} cases written to {DATA_DIR} / {TRUTH_DIR}")


if __name__ == "__main__":
    main()
