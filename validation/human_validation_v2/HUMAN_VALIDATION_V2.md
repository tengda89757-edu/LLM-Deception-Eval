# Human Validation V2 Rerun Workspace

This folder contains the fresh independent tactic-annotation rerun.

## Active Input

- `human_validation_V2_N360_random_annotation_sheet.csv`
- `human_validation_V2_N360_random_manifest.json`

## Rerun Outputs

- `rerun_annotation_claude_code.csv`
- `rerun_annotation_gemini_cli.csv`
- `rerun_annotation_disagreements.csv`
- `rerun_annotation_adjudicated.csv`
- `rerun_annotation_summary.json`

## Rerun Summary

- Generated validation rows annotated: 62
- Adjudicated strict-deception positive: 12/62 = 19.4%
- Adjudicated broad-codebook positive: 20/62 = 32.3%
- Validation-sample target shift among generated rows: 42/62 = 67.7%
- Target shift among strict-positive rows: 10/12 = 83.3%
- Target shift among broad-positive rows: 14/20 = 70.0%
- Strict-label pre-adjudication agreement: 53/62 = 85.5%, Cohen's kappa = 0.403
- Broad-label pre-adjudication agreement: 25/62 = 40.3%, Cohen's kappa = 0.040

Interpretation: the strict label is usable only as an adjudicated exploratory validation-sample audit. The broad codebook is boundary-sensitive and should be reported as a sensitivity audit rather than a strict-deception measure or stable tactic classifier.

## Archived Prior V2 Outputs

Prior annotator files, machine key, result summaries, and the previous validation report were moved to:

`revision_supplementary/archive/human_validation_v2_pre_rerun_20260530_105118/`

Do not use archived files while producing the fresh annotations.
