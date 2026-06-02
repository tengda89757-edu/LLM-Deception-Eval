#!/usr/bin/env python3
"""Export unified random-N360 human-validation packets.

V2 uses one reproducible global random sample from the canonical evidence lock.
The same N=360 rows validate core labels; rows that engaged also receive
strict/broad deception-tactic labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEXT_COLUMNS = [
    "annotation_id",
    "row_id",
    "scenario_id",
    "topic",
    "topic_group",
    "compact_domain",
    "target_option_id",
    "target_option_text",
    "deceptive_goal",
    "objective_goal",
    "context_document",
    "decision_options",
    "gate_raw_text",
    "attacker_intervention_text",
    "baseline_raw_text",
    "neutral_expert_raw_text",
    "neutral_defender_raw_text",
    "target_aware_defender_raw_text",
    "tactic_scope",
    "context_document_word_count",
    "attacker_response_len_tokens",
    "ScenarioComplexity_z",
]

GENERIC_LABEL_COLUMNS = [
    "refusal_label_5way",
    "engagement_binary",
    "baseline_option_id",
    "neutral_option_id",
    "target_aware_option_id",
    "strict_explicit_falsehood",
    "strict_fabricated_source_or_statistic",
    "strict_false_attribution",
    "broad_salient_omission",
    "broad_misleading_framing",
    "persuasive_but_true_only",
    "v_strict_binary",
    "v_broad_binary",
    "confidence_1_to_5",
    "uncertainty_flag",
    "evidence_spans_or_notes",
    "boundary_notes",
]

MACHINE_KEY_COLUMNS = [
    "annotation_id",
    "row_id",
    "scenario_id",
    "topic",
    "topic_group",
    "compact_domain",
    "gate_iv_arm",
    "engagement_binary",
    "refusal_label_5way",
    "attacker_family",
    "defender_family",
    "target_option_id",
    "baseline_option_id",
    "control_option_id",
    "neutral_control_option_id",
    "deception_option_id",
    "deception_success_binary",
    "tactic_tags_model_final",
    "tactic_count_model_final",
    "context_document_word_count",
    "attacker_response_len_tokens",
    "attacker_capability_z",
    "defender_capability_z",
    "ScenarioComplexity_z",
    "refusal_confidence",
    "attacker_response_hash",
    "deception_defender_response_hash",
]


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if value is None:
        return False
    return str(value).strip() in {"1", "1.0", "true", "True", "TRUE"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def choose_text(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def build_base_rows(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sample, start=1):
        engaged = truthy(row.get("engagement_binary"))
        out_rows.append(
            {
                "annotation_id": f"HV2_360_{index:04d}",
                "row_id": row.get("row_id", ""),
                "scenario_id": row.get("scenario_id", ""),
                "topic": row.get("topic", ""),
                "topic_group": row.get("topic_group", ""),
                "compact_domain": row.get("compact_domain", ""),
                "target_option_id": row.get("target_option_id", ""),
                "target_option_text": row.get("target_option_text", ""),
                "deceptive_goal": row.get("deceptive_goal", ""),
                "objective_goal": row.get("objective_goal", ""),
                "context_document": row.get("context_document", ""),
                "decision_options": row.get("decision_options", ""),
                "gate_raw_text": row.get("gate_raw_text", ""),
                "attacker_intervention_text": row.get("attacker_raw_text", "") if engaged else "",
                "baseline_raw_text": row.get("baseline_raw_text", ""),
                "neutral_expert_raw_text": choose_text(row, "neutral_control_expert_raw_text", "control_expert_raw_text"),
                "neutral_defender_raw_text": choose_text(row, "neutral_control_defender_raw_text", "control_defender_raw_text"),
                "target_aware_defender_raw_text": row.get("deception_defender_raw_text", "") if engaged else "",
                "tactic_scope": "LABEL_GENERATED_INTERVENTION" if engaged else "NOT_APPLICABLE_NO_GENERATED_INTERVENTION",
                "context_document_word_count": row.get("context_document_word_count", ""),
                "attacker_response_len_tokens": row.get("attacker_response_len_tokens", "") if engaged else "",
                "ScenarioComplexity_z": row.get("ScenarioComplexity_z", ""),
            }
        )
    return out_rows


def add_generic_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        for column in GENERIC_LABEL_COLUMNS:
            new[column] = ""
        out_rows.append(new)
    return out_rows


def add_prefixed_labels(rows: list[dict[str, Any]], prefix: str) -> tuple[list[str], list[dict[str, Any]]]:
    label_columns = [f"{prefix}_{column}" for column in GENERIC_LABEL_COLUMNS]
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        for column in label_columns:
            new[column] = ""
        out_rows.append(new)
    return TEXT_COLUMNS + label_columns, out_rows


def build_machine_key(base_rows: list[dict[str, Any]], source_by_row_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    key_rows: list[dict[str, Any]] = []
    for base in base_rows:
        row = source_by_row_id[base["row_id"]]
        key_rows.append(
            {
                "annotation_id": base["annotation_id"],
                "row_id": row.get("row_id", ""),
                "scenario_id": row.get("scenario_id", ""),
                "topic": row.get("topic", ""),
                "topic_group": row.get("topic_group", ""),
                "compact_domain": row.get("compact_domain", ""),
                "gate_iv_arm": row.get("gate_iv_arm", ""),
                "engagement_binary": row.get("engagement_binary", ""),
                "refusal_label_5way": row.get("refusal_label_5way", ""),
                "attacker_family": row.get("attacker_family", ""),
                "defender_family": row.get("defender_family", ""),
                "target_option_id": row.get("target_option_id", ""),
                "baseline_option_id": row.get("baseline_option_id", ""),
                "control_option_id": row.get("control_option_id", ""),
                "neutral_control_option_id": row.get("neutral_control_option_id", ""),
                "deception_option_id": row.get("deception_option_id", ""),
                "deception_success_binary": row.get("deception_success_binary", ""),
                "tactic_tags_model_final": row.get("tactic_tags_model_final", ""),
                "tactic_count_model_final": row.get("tactic_count_model_final", ""),
                "context_document_word_count": row.get("context_document_word_count", ""),
                "attacker_response_len_tokens": row.get("attacker_response_len_tokens", ""),
                "attacker_capability_z": row.get("attacker_capability_z", ""),
                "defender_capability_z": row.get("defender_capability_z", ""),
                "ScenarioComplexity_z": row.get("ScenarioComplexity_z", ""),
                "refusal_confidence": row.get("refusal_confidence", ""),
                "attacker_response_hash": row.get("attacker_response_hash", ""),
                "deception_defender_response_hash": row.get("deception_defender_response_hash", ""),
            }
        )
    return key_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-rows",
        type=Path,
        default=Path("outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/analysis_rows.jsonl"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("revision_supplementary/human_validation_v2"))
    parser.add_argument("--sample-size", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.analysis_rows)
    if args.sample_size > len(rows):
        raise SystemExit("sample-size exceeds available rows")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.sample_size)
    sample.sort(key=lambda row: row.get("row_id", ""))

    base_rows = build_base_rows(sample)
    source_by_row_id = {row["row_id"]: row for row in rows}
    generic_rows = add_generic_labels(base_rows)
    annotator1_columns, annotator1_rows = add_prefixed_labels(base_rows, "annotator1")
    annotator2_columns, annotator2_rows = add_prefixed_labels(base_rows, "annotator2")
    machine_key_rows = build_machine_key(base_rows, source_by_row_id)

    annotation_path = args.outdir / "human_validation_V2_N360_random_annotation_sheet.csv"
    annotator1_path = args.outdir / "human_validation_V2_N360_random_annotator1.csv"
    annotator2_path = args.outdir / "human_validation_V2_N360_random_annotator2.csv"
    machine_key_path = args.outdir / "human_validation_V2_N360_random_machine_key.csv"
    manifest_path = args.outdir / "human_validation_V2_N360_random_manifest.json"

    write_csv(annotation_path, TEXT_COLUMNS + GENERIC_LABEL_COLUMNS, generic_rows)
    write_csv(annotator1_path, annotator1_columns, annotator1_rows)
    write_csv(annotator2_path, annotator2_columns, annotator2_rows)
    write_csv(machine_key_path, MACHINE_KEY_COLUMNS, machine_key_rows)

    engaged_sample = [row for row in sample if truthy(row.get("engagement_binary"))]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis_rows": str(args.analysis_rows),
        "sampling": {
            "method": "simple_random_sample_without_replacement",
            "sample_size": args.sample_size,
            "seed": args.seed,
            "sample_frame_n": len(rows),
        },
        "row_count_validation_sample": len(sample),
        "row_count_engaged_in_validation_sample": len(engaged_sample),
        "row_count_target_shift_among_engaged_validation_sample": sum(
            1 for row in engaged_sample if truthy(row.get("deception_success_binary"))
        ),
        "gate_distribution_validation_sample": dict(Counter(row.get("gate_iv_arm", "") for row in sample)),
        "gate_distribution_engaged_validation_sample": dict(
            Counter(row.get("gate_iv_arm", "") for row in engaged_sample)
        ),
        "domain_distribution_validation_sample": dict(Counter(row.get("compact_domain", "") for row in sample)),
        "annotation_files": [
            str(annotation_path),
            str(annotator1_path),
            str(annotator2_path),
        ],
        "analyst_only_machine_key": str(machine_key_path),
        "annotator_blinding": {
            "included": [
                "context_document",
                "decision_options",
                "target_option_id",
                "target_option_text",
                "deceptive_goal",
                "gate_raw_text",
                "baseline_raw_text",
                "neutral_defender_raw_text",
                "target_aware_defender_raw_text for engaged rows",
                "attacker_intervention_text for engaged rows",
            ],
            "excluded": [
                "gate_iv_arm",
                "attacker_family",
                "defender_family",
                "machine-derived labels",
                "deception_success_binary",
                "machine tactic labels",
            ],
        },
        "status": "annotation_template_exported_not_completed",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
