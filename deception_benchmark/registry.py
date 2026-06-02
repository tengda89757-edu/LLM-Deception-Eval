from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deception_benchmark.config import (
    ALL_MODELS,
    BRANCH_INTERACTIONS_CSV,
    BRANCH_INTERACTIONS_JSONL,
    DEFAULT_SEED,
    JOINED_ROWS_CSV,
    JOINED_ROWS_JSONL,
    MANUAL_DIR,
    MODEL_CAPABILITIES_PATH,
    OPTION_IDS,
    RAW_SCENARIO_DIR,
    RUN_TABLE_CSV,
    RUN_TABLE_JSONL,
    SCENARIO_COMPLEXITY_PATH,
    SCENARIO_METADATA_PATH,
    SCENARIO_REGISTRY_CSV,
    SCENARIO_REGISTRY_JSONL,
    ensure_directories,
)


SCENARIO_METADATA_FIELDS = [
    "scenario_id",
    "source_type",
    "source_reference",
    "gold_baseline_option_id",
    "risk_level",
    "information_asymmetry_level",
    "time_pressure_level",
    "notes",
]


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_scenarios() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(RAW_SCENARIO_DIR.rglob("*.json")):
        payload = _load_json(path)
        payload["_source_path"] = path.relative_to(RAW_SCENARIO_DIR.parent).as_posix()
        records.append(payload)
    records.sort(key=lambda item: item["scenario_id"])
    if len(records) != 53:
        raise ValueError(f"Expected 53 raw scenarios, found {len(records)}")
    return records


def _ensure_scenario_metadata_template(raw_scenarios: list[dict[str, Any]]) -> None:
    if SCENARIO_METADATA_PATH.exists():
        return
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    with SCENARIO_METADATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCENARIO_METADATA_FIELDS)
        writer.writeheader()
        for scenario in raw_scenarios:
            writer.writerow(
                {
                    "scenario_id": scenario["scenario_id"],
                    "source_type": "",
                    "source_reference": "",
                    "gold_baseline_option_id": "",
                    "risk_level": "",
                    "information_asymmetry_level": "",
                    "time_pressure_level": "",
                    "notes": "",
                }
            )


def _ensure_model_capability_template() -> None:
    if MODEL_CAPABILITIES_PATH.exists():
        return
    with MODEL_CAPABILITIES_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["family", "model_slug", "provider", "capability_score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in ALL_MODELS:
            writer.writerow(
                {
                    "family": model["family"],
                    "model_slug": model["model_slug"],
                    "provider": model["provider"],
                    "capability_score": "",
                }
            )


def _ensure_scenario_complexity_template(raw_scenarios: list[dict[str, Any]]) -> None:
    if SCENARIO_COMPLEXITY_PATH.exists():
        return
    with SCENARIO_COMPLEXITY_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scenario_id",
            "evidence_ambiguity",
            "causal_entanglement",
            "option_discriminability",
            "counterevidence_salience",
            "deception_plausibility",
            "ecological_realism",
            "gold_baseline_option_valid",
            "ScenarioComplexity_raw",
            "ScenarioComplexity_z",
            "gold_baseline_option_human_vote",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in raw_scenarios:
            writer.writerow({"scenario_id": scenario["scenario_id"]})


def load_manual_metadata(raw_scenarios: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    _ensure_scenario_metadata_template(raw_scenarios)
    _ensure_model_capability_template()
    _ensure_scenario_complexity_template(raw_scenarios)
    with SCENARIO_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["scenario_id"]: row for row in rows}


def _option_id_for_index(index: int) -> str:
    zero_based = index - 1
    if zero_based < 0 or zero_based >= len(OPTION_IDS):
        raise ValueError(f"Unsupported option index: {index}")
    return OPTION_IDS[zero_based]


def _option_records(options: list[str]) -> list[dict[str, str]]:
    if len(options) > len(OPTION_IDS):
        raise ValueError(f"Unsupported option count: {len(options)}")
    return [
        {"option_id": OPTION_IDS[idx], "option_text": option}
        for idx, option in enumerate(options)
    ]


def build_scenario_registry(
    raw_scenarios: list[dict[str, Any]], metadata: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in raw_scenarios:
        meta = metadata.get(raw["scenario_id"], {})
        option_records = _option_records(raw["decision_options"])
        record = {
            "scenario_id": raw["scenario_id"],
            "legacy_category": raw["legacy_category"],
            "topic_group": raw["legacy_category"],
            "compact_domain": raw["compact_domain"],
            "compact_subdomain": raw["compact_subdomain"],
            "topic": raw["topic"],
            "source_type": meta.get("source_type") or None,
            "source_reference": meta.get("source_reference") or None,
            "context_document": raw["context_document"],
            "context_document_word_count": len(raw["context_document"].split()),
            "deceptive_goal": raw["deceptive_goal"],
            "objective_goal": "Provide balanced and objective analysis",
            "decision_options": option_records,
            "decision_option_count": len(option_records),
            "target_option_id": _option_id_for_index(raw["target_option"]["index"]),
            "target_option_text": raw["target_option"]["text"],
            "gold_baseline_option_id": meta.get("gold_baseline_option_id") or None,
            "risk_level": meta.get("risk_level") or None,
            "information_asymmetry_level": meta.get("information_asymmetry_level") or None,
            "time_pressure_level": meta.get("time_pressure_level") or None,
            "source_path": raw["_source_path"],
            "seed": DEFAULT_SEED,
        }
        records.append(record)
    return records


def build_joined_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for attacker in ALL_MODELS:
            for defender in ALL_MODELS:
                if defender["family"] == attacker["family"]:
                    continue
                rows.append(
                    {
                        "row_id": f"{scenario['scenario_id']}|{attacker['family']}|{defender['family']}",
                        "scenario_id": scenario["scenario_id"],
                        "topic_group": scenario["topic_group"],
                        "topic": scenario["topic"],
                        "legacy_category": scenario["legacy_category"],
                        "compact_domain": scenario["compact_domain"],
                        "source_type": scenario["source_type"],
                        "target_option_id": scenario["target_option_id"],
                        "gold_baseline_option_id": scenario["gold_baseline_option_id"],
                        "context_document_word_count": scenario["context_document_word_count"],
                        "risk_level": scenario["risk_level"],
                        "information_asymmetry_level": scenario["information_asymmetry_level"],
                        "time_pressure_level": scenario["time_pressure_level"],
                        "attacker_family": attacker["family"],
                        "attacker_model": attacker["model_slug"],
                        "attacker_provider": attacker["provider"],
                        "defender_family": defender["family"],
                        "defender_model": defender["model_slug"],
                        "defender_provider": defender["provider"],
                    }
                )
    return rows


def build_branch_interactions(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in joined_rows:
        for branch in ("baseline", "deception", "control"):
            rows.append(
                {
                    **row,
                    "run_id": f"{row['row_id']}|{branch}",
                    "branch": branch,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            serialized = {
                key: _json_dump(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_registry_outputs(
    scenarios: list[dict[str, Any]],
    joined_rows: list[dict[str, Any]],
    branch_interactions: list[dict[str, Any]],
) -> None:
    ensure_directories()
    _write_csv(SCENARIO_REGISTRY_CSV, scenarios)
    _write_jsonl(SCENARIO_REGISTRY_JSONL, scenarios)
    _write_csv(JOINED_ROWS_CSV, joined_rows)
    _write_jsonl(JOINED_ROWS_JSONL, joined_rows)
    _write_csv(BRANCH_INTERACTIONS_CSV, branch_interactions)
    _write_jsonl(BRANCH_INTERACTIONS_JSONL, branch_interactions)
    _write_csv(RUN_TABLE_CSV, branch_interactions)
    _write_jsonl(RUN_TABLE_JSONL, branch_interactions)

