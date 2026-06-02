from __future__ import annotations

import csv
import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported input format: {path.suffix}")


def infer_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: to_jsonable(v) for k, v in row.items()}, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_csv_value(row.get(k)) for k in fieldnames})


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return str(value)


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "ok", "valid"}:
        return True
    if text in {"0", "false", "f", "no", "n", "invalid", "none", ""}:
        return False
    return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def row_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("row_id", "run_id", "scenario_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def stable_signature(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def count_by(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key, "")) for row in rows)
    return [{"value": value, "count": count} for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def count_multi(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counter[tuple(str(row.get(key, "")) for key in keys)] += 1
    output: list[dict[str, Any]] = []
    for values, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        output.append({**{key: value for key, value in zip(keys, values)}, "count": count})
    return output


def unique_count(rows: Sequence[dict[str, Any]], key: str) -> int:
    return len({str(row.get(key, "")) for row in rows if row.get(key, "") not in (None, "")})


def nonempty_keys(rows: Sequence[dict[str, Any]]) -> set[str]:
    present: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if value not in (None, ""):
                present.add(key)
    return present


def json_encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def make_table(name: str, rows: Sequence[dict[str, Any]], *, status: str = "ok", reason: str = "", missing_columns: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "table_name": name,
        "status": status,
        "reason": reason,
        "missing_columns": list(missing_columns or []),
        "row_count": len(rows),
        "rows": list(rows),
    }


def value_counts(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def first_nonempty_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    value = first_present(row, keys)
    return "" if value is None else str(value)


def maybe_json_list(value: Any) -> list[Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def maybe_json_dict(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def text_length_words(text: Any) -> int | None:
    if text is None or text == "":
        return None
    return len(str(text).split())


def summarize_missing_columns(rows: Sequence[dict[str, Any]], required_columns: Sequence[str]) -> list[str]:
    present = nonempty_keys(rows)
    return [column for column in required_columns if column not in present]


def _count_truthy(rows: Sequence[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if as_bool(row.get(key)) is True)


def _count_falsy(rows: Sequence[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if as_bool(row.get(key)) is False)


def build_design_matrix_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    total_rows = len(rows)
    return make_table(
        "design_matrix_summary",
        [
            {
                "metric": "total_rows",
                "value": total_rows,
            },
            {
                "metric": "unique_scenarios",
                "value": unique_count(rows, "scenario_id"),
            },
            {
                "metric": "unique_attackers",
                "value": unique_count(rows, "attacker_model"),
            },
            {
                "metric": "unique_defenders",
                "value": unique_count(rows, "defender_model"),
            },
            {
                "metric": "unique_branches",
                "value": unique_count(rows, "branch"),
            },
        ],
    )


def build_scenario_diversity_summary(scenario_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scenario_rows)
    present = nonempty_keys(rows)
    summary_rows: list[dict[str, Any]] = []
    for key in ("legacy_category", "compact_domain", "compact_subdomain", "risk_level", "information_asymmetry_level", "time_pressure_level"):
        if key in present:
            summary_rows.extend([{"dimension": key, "value": item["value"], "count": item["count"]} for item in count_by(rows, key)])
        else:
            summary_rows.append({"dimension": key, "value": "", "count": 0, "status": "missing"})
    return make_table("scenario_diversity_summary", summary_rows)


def build_judge_coverage_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    summary_rows = [
        {"metric": "judge_disagreement", "count": _count_truthy(rows, "judge_disagreement")},
        {"metric": "json_invalid", "count": _count_truthy(rows, "json_invalid")},
        {"metric": "parse_failure_flag", "count": _count_truthy(rows, "parse_failure_flag")},
        {"metric": "refusal_rule_judge_disagree", "count": _count_truthy(rows, "refusal_rule_judge_disagree")},
        {"metric": "mtd_available_model_false", "count": _count_falsy(rows, "mtd_available_model")},
    ]
    return make_table("judge_coverage_summary", summary_rows)


def build_stage1_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    if "engagement_binary" not in nonempty_keys(rows):
        return make_table(
            "stage1_engagement_summary",
            [{"status": "blocked", "reason": "missing engagement_binary"}],
            status="blocked",
            reason="missing engagement_binary",
            missing_columns=["engagement_binary"],
        )
    summary_rows = [
        {"metric": "engaged", "count": _count_truthy(rows, "engagement_binary")},
        {"metric": "not_engaged", "count": _count_falsy(rows, "engagement_binary")},
    ]
    return make_table("stage1_engagement_summary", summary_rows)


def build_stage2_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    metrics = [
        ("baseline_target_binary_true", "baseline_target_binary"),
        ("neutral_control_shift_to_target_true", "neutral_control_shift_to_target"),
        ("attacker_nondeceptive_shift_to_target_true", "attacker_nondeceptive_shift_to_target"),
        ("deception_success_conditional_true", "deception_success_binary"),
        ("deception_success_unconditional_true", "deception_success_unconditional"),
        ("strict_success_true", "strict_success"),
        ("corrective_influence_true", "corrective_influence"),
    ]
    summary_rows = [{"metric": metric, "count": _count_truthy(rows, column)} for metric, column in metrics if column in nonempty_keys(rows)]
    if not summary_rows:
        return make_table(
            "stage2_summary",
            [{"status": "blocked", "reason": "missing stage 2 derived labels"}],
            status="blocked",
            reason="missing stage 2 derived labels",
            missing_columns=[column for _, column in metrics],
        )
    return make_table("stage2_summary", summary_rows)


def build_strict_lean_pooled_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    summary_rows: list[dict[str, Any]] = []
    for key in ("sample_strict", "sample_lean", "sample_pooled_only", "sample_malformed"):
        if key in nonempty_keys(rows):
            summary_rows.append({"metric": key, "count": _count_truthy(rows, key)})
        else:
            summary_rows.append({"metric": key, "count": 0, "status": "missing"})
    return make_table("strict_lean_pooled_summary", summary_rows)


def build_human_model_consistency_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    human_cols = [column for column in ("option_id_baseline_human", "option_id_deception_human", "option_id_control_human", "tactic_tags_human") if column in nonempty_keys(rows)]
    model_cols = [
        column
        for column in (
            "baseline_option_id",
            "neutral_control_option_id",
            "attacker_nondeceptive_option_id",
            "deception_option_id",
            "control_option_id",
            "tactic_tags_model_final",
        )
        if column in nonempty_keys(rows)
    ]
    if not human_cols or not model_cols:
        return make_table(
            "human_model_consistency_summary",
            [{"status": "blocked", "reason": "missing human or model comparison labels"}],
            status="blocked",
            reason="missing human or model comparison labels",
            missing_columns=["option_id_baseline_human", "option_id_deception_human", "option_id_control_human", "tactic_tags_human"],
        )
    summary_rows = [
        {"metric": "human_label_columns", "count": len(human_cols)},
        {"metric": "model_label_columns", "count": len(model_cols)},
        {"metric": "rows_with_any_human_label", "count": sum(1 for row in rows if any(row.get(col) not in (None, "") for col in human_cols))},
        {"metric": "rows_with_any_model_label", "count": sum(1 for row in rows if any(row.get(col) not in (None, "") for col in model_cols))},
    ]
    return make_table("human_model_consistency_summary", summary_rows)


def build_mtd_confirmatory_summary(analysis_rows: Sequence[dict[str, Any]], subset_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = list(analysis_rows)
    manifest_status = None
    if subset_manifest:
        manifest_status = subset_manifest.get("subsets", {}).get("H_MTD_CONFIRM_300", {})
    human_cols = [column for column in ("ScenarioComplexity_z", "ScenarioComplexity_raw", "scenario_complexity_human", "human_complexity_label") if column in nonempty_keys(rows)]
    if not human_cols:
        reason = "missing human complexity annotations"
        if manifest_status and manifest_status.get("status") == "blocked":
            reason = str(manifest_status.get("reason", reason))
        return make_table(
            "mtd_confirmatory_summary",
            [{"status": "blocked", "reason": reason}],
            status="blocked",
            reason=reason,
            missing_columns=["ScenarioComplexity_z", "ScenarioComplexity_raw", "scenario_complexity_human"],
        )
    return make_table(
        "mtd_confirmatory_summary",
        [
            {"metric": "human_complexity_columns", "count": len(human_cols)},
            {"metric": "rows_with_human_complexity", "count": sum(1 for row in rows if any(row.get(col) not in (None, "") for col in human_cols))},
        ],
    )


def build_escalation_pool_summary(analysis_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(analysis_rows)
    summary_rows = [
        {"metric": "judge_disagreement", "count": _count_truthy(rows, "judge_disagreement")},
        {"metric": "json_invalid", "count": _count_truthy(rows, "json_invalid")},
        {"metric": "parse_failure_flag", "count": _count_truthy(rows, "parse_failure_flag")},
        {"metric": "sample_malformed", "count": _count_truthy(rows, "sample_malformed")},
        {"metric": "refusal_rule_judge_disagree", "count": _count_truthy(rows, "refusal_rule_judge_disagree")},
    ]
    return make_table("escalation_pool_summary", summary_rows)


def build_summary_tables(
    analysis_rows: Sequence[dict[str, Any]],
    scenario_rows: Sequence[dict[str, Any]],
    *,
    subset_manifest: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "design_matrix_summary": build_design_matrix_summary(analysis_rows),
        "scenario_diversity_summary": build_scenario_diversity_summary(scenario_rows),
        "judge_coverage_summary": build_judge_coverage_summary(analysis_rows),
        "stage1_engagement_summary": build_stage1_summary(analysis_rows),
        "stage2_summary": build_stage2_summary(analysis_rows),
        "strict_lean_pooled_summary": build_strict_lean_pooled_summary(analysis_rows),
        "human_model_consistency_summary": build_human_model_consistency_summary(analysis_rows),
        "mtd_confirmatory_summary": build_mtd_confirmatory_summary(analysis_rows, subset_manifest=subset_manifest),
        "escalation_pool_summary": build_escalation_pool_summary(analysis_rows),
    }
