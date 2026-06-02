#!/usr/bin/env python3
"""Summarize V2 human validation agreement and audit results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CORE_FIELDS = [
    ("refusal_label_5way", "all"),
    ("engagement_binary", "all"),
    ("baseline_option_id", "all"),
    ("neutral_option_id", "all"),
    ("target_aware_option_id", "engaged"),
]

TACTIC_FIELDS = [
    "strict_explicit_falsehood",
    "strict_fabricated_source_or_statistic",
    "strict_false_attribution",
    "broad_salient_omission",
    "broad_misleading_framing",
    "persuasive_but_true_only",
    "v_strict_binary",
    "v_broad_binary",
]

MACHINE_FIELD_MAP = {
    "refusal_label_5way": "refusal_label_5way",
    "engagement_binary": "engagement_binary",
    "baseline_option_id": "baseline_option_id",
    "neutral_option_id": "neutral_control_option_id",
    "target_aware_option_id": "deception_option_id",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def norm(value: Any, field: str | None = None) -> str:
    text = "" if value is None else str(value).strip()
    if text in {"1.0", "1", "true", "True", "TRUE"}:
        return "1"
    if text in {"0.0", "0", "false", "False", "FALSE"}:
        return "0"
    if field == "refusal_label_5way":
        aliases = {
            "REFUSE_SAFETY": "REFUSAL_SAFETY",
            "NOT_REFUSED": "FULL_ENGAGEMENT",
        }
        return aliases.get(text, text)
    return text


def is_engaged(row: dict[str, str]) -> bool:
    return norm(row.get("engagement_binary", "")) == "1"


def cohen_kappa(labels1: list[str], labels2: list[str]) -> float | None:
    if not labels1 or len(labels1) != len(labels2):
        return None
    n = len(labels1)
    observed = sum(a == b for a, b in zip(labels1, labels2)) / n
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    cats = set(c1) | set(c2)
    expected = sum((c1[c] / n) * (c2[c] / n) for c in cats)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def metric_row(field: str, scope: str, values1: list[str], values2: list[str]) -> dict[str, Any]:
    n = len(values1)
    agreed = sum(a == b for a, b in zip(values1, values2))
    kappa = cohen_kappa(values1, values2)
    return {
        "field": field,
        "scope": scope,
        "n": n,
        "raw_agreement": round(agreed / n, 4) if n else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "annotator1_distribution": dict(Counter(values1)),
        "annotator2_distribution": dict(Counter(values2)),
    }


def positive_rate(values: list[str]) -> dict[str, Any]:
    n = len(values)
    positives = sum(v == "1" for v in values)
    return {
        "n": n,
        "positive": positives,
        "rate": round(positives / n, 4) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("revision_supplementary/human_validation_v2"))
    args = parser.parse_args()

    base = args.base_dir
    a1 = read_csv(base / "human_validation_V2_N360_random_annotator1.csv")
    a2 = read_csv(base / "human_validation_V2_N360_random_annotator2.csv")
    key = read_csv(base / "human_validation_V2_N360_random_machine_key.csv")
    outdir = base / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    by_id_1 = {row["annotation_id"]: row for row in a1}
    by_id_2 = {row["annotation_id"]: row for row in a2}
    by_id_key = {row["annotation_id"]: row for row in key}
    ids = sorted(set(by_id_1) & set(by_id_2) & set(by_id_key))
    manifest_path = base / "human_validation_V2_N360_random_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    agreement_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    machine_rows: list[dict[str, Any]] = []

    for field, scope in CORE_FIELDS:
        scoped_ids = [i for i in ids if scope == "all" or is_engaged(by_id_key[i])]
        values1 = [norm(by_id_1[i].get(f"annotator1_{field}", ""), field) for i in scoped_ids]
        values2 = [norm(by_id_2[i].get(f"annotator2_{field}", ""), field) for i in scoped_ids]
        nonempty = [(i, v1, v2) for i, v1, v2 in zip(scoped_ids, values1, values2) if v1 != "" and v2 != ""]
        agreement_rows.append(metric_row(field, scope, [x[1] for x in nonempty], [x[2] for x in nonempty]))

        for i, v1, v2 in nonempty:
            if v1 != v2:
                disagreement_rows.append(
                    {
                        "annotation_id": i,
                        "row_id": by_id_key[i].get("row_id", ""),
                        "field": field,
                        "scope": scope,
                        "annotator1": v1,
                        "annotator2": v2,
                        "machine": norm(by_id_key[i].get(MACHINE_FIELD_MAP[field], ""), field),
                    }
                )

        machine_field = MACHINE_FIELD_MAP[field]
        for prefix, annotator_rows in [("annotator1", by_id_1), ("annotator2", by_id_2)]:
            human_vals = []
            machine_vals = []
            for i in scoped_ids:
                h = norm(annotator_rows[i].get(f"{prefix}_{field}", ""), field)
                m = norm(by_id_key[i].get(machine_field, ""), field)
                if h != "" and m != "":
                    human_vals.append(h)
                    machine_vals.append(m)
            machine_rows.append(metric_row(f"{prefix}_vs_machine__{field}", scope, human_vals, machine_vals))

    engaged_ids = [i for i in ids if is_engaged(by_id_key[i])]
    for field in TACTIC_FIELDS:
        values1 = [norm(by_id_1[i].get(f"annotator1_{field}", ""), field) for i in engaged_ids]
        values2 = [norm(by_id_2[i].get(f"annotator2_{field}", ""), field) for i in engaged_ids]
        nonempty = [(i, v1, v2) for i, v1, v2 in zip(engaged_ids, values1, values2) if v1 != "" and v2 != ""]
        agreement_rows.append(metric_row(field, "sampled_engaged", [x[1] for x in nonempty], [x[2] for x in nonempty]))
        for i, v1, v2 in nonempty:
            if v1 != v2:
                disagreement_rows.append(
                    {
                        "annotation_id": i,
                        "row_id": by_id_key[i].get("row_id", ""),
                        "field": field,
                        "scope": "sampled_engaged",
                        "annotator1": v1,
                        "annotator2": v2,
                        "machine": "",
                    }
                )

    # Consensus-positive only where annotators agree. Disagreements remain adjudication-pending.
    consensus = {}
    for field in TACTIC_FIELDS:
        vals = []
        for i in engaged_ids:
            v1 = norm(by_id_1[i].get(f"annotator1_{field}", ""), field)
            v2 = norm(by_id_2[i].get(f"annotator2_{field}", ""), field)
            vals.append(v1 if v1 == v2 and v1 != "" else "DISAGREE_OR_MISSING")
        consensus[field] = dict(Counter(vals))

    target_shift_ids = [i for i in engaged_ids if norm(by_id_key[i].get("deception_success_binary", "")) == "1"]
    strict_consensus_positive = [
        i
        for i in engaged_ids
        if norm(by_id_1[i].get("annotator1_v_strict_binary", ""), "v_strict_binary")
        == norm(by_id_2[i].get("annotator2_v_strict_binary", ""), "v_strict_binary")
        == "1"
    ]
    broad_consensus_positive = [
        i
        for i in engaged_ids
        if norm(by_id_1[i].get("annotator1_v_broad_binary", ""), "v_broad_binary")
        == norm(by_id_2[i].get("annotator2_v_broad_binary", ""), "v_broad_binary")
        == "1"
    ]

    summary = {
        "sampling": manifest.get("sampling", {}),
        "counts": {
            "rows_total": len(ids),
            "rows_engaged_sample": len(engaged_ids),
            "rows_target_shift_among_engaged_sample": len(target_shift_ids),
            "field_level_core_disagreements": sum(1 for row in disagreement_rows if row["scope"] != "sampled_engaged"),
            "field_level_tactic_disagreements": sum(1 for row in disagreement_rows if row["scope"] == "sampled_engaged"),
            "unique_core_disagreement_rows": len(
                {row["annotation_id"] for row in disagreement_rows if row["scope"] != "sampled_engaged"}
            ),
            "unique_tactic_disagreement_rows": len(
                {row["annotation_id"] for row in disagreement_rows if row["scope"] == "sampled_engaged"}
            ),
        },
        "agreement": agreement_rows,
        "human_vs_machine": machine_rows,
        "tactic_consensus_distribution": consensus,
        "tactic_rates_by_annotator": {
            "annotator1_v_strict": positive_rate(
                [norm(by_id_1[i].get("annotator1_v_strict_binary", ""), "v_strict_binary") for i in engaged_ids]
            ),
            "annotator2_v_strict": positive_rate(
                [norm(by_id_2[i].get("annotator2_v_strict_binary", ""), "v_strict_binary") for i in engaged_ids]
            ),
            "annotator1_v_broad": positive_rate(
                [norm(by_id_1[i].get("annotator1_v_broad_binary", ""), "v_broad_binary") for i in engaged_ids]
            ),
            "annotator2_v_broad": positive_rate(
                [norm(by_id_2[i].get("annotator2_v_broad_binary", ""), "v_broad_binary") for i in engaged_ids]
            ),
            "consensus_positive_strict": positive_rate(["1" if i in strict_consensus_positive else "0" for i in engaged_ids]),
            "consensus_positive_broad": positive_rate(["1" if i in broad_consensus_positive else "0" for i in engaged_ids]),
        },
        "target_shift_join": {
            "strict_consensus_positive_and_target_shift": {
                "n": len(strict_consensus_positive),
                "target_shift": sum(1 for i in strict_consensus_positive if i in set(target_shift_ids)),
            },
            "broad_consensus_positive_and_target_shift": {
                "n": len(broad_consensus_positive),
                "target_shift": sum(1 for i in broad_consensus_positive if i in set(target_shift_ids)),
            },
        },
        "status": "adjudication_pending" if disagreement_rows else "no_disagreements_detected",
    }

    (outdir / "human_validation_V2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        outdir / "human_validation_V2_agreement_summary.csv",
        agreement_rows + machine_rows,
        ["field", "scope", "n", "raw_agreement", "cohen_kappa", "annotator1_distribution", "annotator2_distribution"],
    )
    write_csv(
        outdir / "human_validation_V2_disagreements.csv",
        disagreement_rows,
        ["annotation_id", "row_id", "field", "scope", "annotator1", "annotator2", "machine"],
    )
    adjudication_rows: list[dict[str, Any]] = []
    disagreement_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in disagreement_rows:
        disagreement_by_id.setdefault(row["annotation_id"], []).append(row)
    for annotation_id, rows_for_id in sorted(disagreement_by_id.items()):
        key_row = by_id_key[annotation_id]
        a1_row = by_id_1[annotation_id]
        a2_row = by_id_2[annotation_id]
        adjudication_rows.append(
            {
                "annotation_id": annotation_id,
                "row_id": key_row.get("row_id", ""),
                "scope": ";".join(sorted({row["scope"] for row in rows_for_id})),
                "disagreed_fields": ";".join(row["field"] for row in rows_for_id),
                "machine_refusal_label_5way": norm(key_row.get("refusal_label_5way", ""), "refusal_label_5way"),
                "machine_engagement_binary": norm(key_row.get("engagement_binary", ""), "engagement_binary"),
                "machine_deception_success_binary": norm(
                    key_row.get("deception_success_binary", ""), "deception_success_binary"
                ),
                "annotator1_refusal_label_5way": norm(
                    a1_row.get("annotator1_refusal_label_5way", ""), "refusal_label_5way"
                ),
                "annotator2_refusal_label_5way": norm(
                    a2_row.get("annotator2_refusal_label_5way", ""), "refusal_label_5way"
                ),
                "annotator1_v_strict_binary": norm(
                    a1_row.get("annotator1_v_strict_binary", ""), "v_strict_binary"
                ),
                "annotator2_v_strict_binary": norm(
                    a2_row.get("annotator2_v_strict_binary", ""), "v_strict_binary"
                ),
                "annotator1_v_broad_binary": norm(
                    a1_row.get("annotator1_v_broad_binary", ""), "v_broad_binary"
                ),
                "annotator2_v_broad_binary": norm(
                    a2_row.get("annotator2_v_broad_binary", ""), "v_broad_binary"
                ),
                "adjudicated_refusal_label_5way": "",
                "adjudicated_v_strict_binary": "",
                "adjudicated_v_broad_binary": "",
                "adjudication_notes": "",
            }
        )
    write_csv(
        outdir / "human_validation_V2_adjudication_needed.csv",
        adjudication_rows,
        [
            "annotation_id",
            "row_id",
            "scope",
            "disagreed_fields",
            "machine_refusal_label_5way",
            "machine_engagement_binary",
            "machine_deception_success_binary",
            "annotator1_refusal_label_5way",
            "annotator2_refusal_label_5way",
            "annotator1_v_strict_binary",
            "annotator2_v_strict_binary",
            "annotator1_v_broad_binary",
            "annotator2_v_broad_binary",
            "adjudicated_refusal_label_5way",
            "adjudicated_v_strict_binary",
            "adjudicated_v_broad_binary",
            "adjudication_notes",
        ],
    )
    print(json.dumps(summary["counts"], indent=2))
    print("status", summary["status"])


if __name__ == "__main__":
    main()
