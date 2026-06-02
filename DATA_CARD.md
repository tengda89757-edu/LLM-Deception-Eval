# Data Card

## Dataset

This package contains 53 synthetic decision scenarios crossed with ordered attacker-defender model pairs for 2,968 evaluation rows.

## Public Evidence Lock

`results/canonical_run/analysis_rows_public.csv` contains 2968 rows and 253 public columns.

Removed columns: 19. See `results/canonical_run/analysis_rows_public_column_policy.json`.

## Intended Uses

- Reproduce denominator-explicit reporting calculations.
- Study generation availability, realized exposure, generated-subset conditional shift, and partial-identification bounds.
- Audit validation-sample tactic-label reliability.
- Run new benchmark instantiations using the published code and scenario registry.

## Out-of-Scope Uses

- Treating generated-subset conditional shift as deployment prevalence.
- Treating tactic labels as population prevalence without separate validation.
- Using scenario prompts or model outputs to optimize manipulative persuasion.

## Privacy and Safety Notes

The scenario data are synthetic benchmark artifacts. The public package removes local paths, secrets, raw provider response files, checkpoint databases, judge logs, and raw generated model dialogue from the canonical analysis table. The scenario registry remains public because it is the benchmark input dataset.
