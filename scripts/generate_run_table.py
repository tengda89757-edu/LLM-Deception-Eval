#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from deception_benchmark.registry import (
    build_branch_interactions,
    build_joined_rows,
    build_scenario_registry,
    load_manual_metadata,
    load_raw_scenarios,
    write_registry_outputs,
)


def main() -> None:
    raw_scenarios = load_raw_scenarios()
    metadata = load_manual_metadata(raw_scenarios)
    scenario_registry = build_scenario_registry(raw_scenarios, metadata)
    joined_rows = build_joined_rows(scenario_registry)
    branch_interactions = build_branch_interactions(joined_rows)
    write_registry_outputs(scenario_registry, joined_rows, branch_interactions)
    summary = {
        "scenario_count": len(scenario_registry),
        "joined_rows": len(joined_rows),
        "branch_interactions": len(branch_interactions),
        "branches": {
            "baseline": len(joined_rows),
            "deception": len(joined_rows),
            "control": len(joined_rows),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
