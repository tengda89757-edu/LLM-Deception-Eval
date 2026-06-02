# Data Dictionary

## Scenario Inputs

- `data/normalized-scenarios/`: scenario JSON files with scenario id, topic, context, target option, decision options, and persuasion objective.
- `data/derived/scenario_registry.csv`: tabular scenario registry.
- `data/derived/joined_rows.csv`: scenario by ordered attacker-defender model-pair design.

## Public Analysis Rows

- `row_id`: scenario and ordered attacker-defender identifier.
- `scenario_id`: scenario identifier.
- `attacker_family`, `defender_family`: model-family labels.
- `gate_iv_arm`: assigned gate framing.
- `engagement_binary`: whether a usable target-aware intervention was generated.
- `deception_success_binary`: generated-subset target shift among generated rows.
- `deception_success_unconditional`: realized exposure on the full denominator.
- `baseline_option_id`, `neutral_control_option_id`, `attacker_nondeceptive_option_id`, `deception_option_id`: parsed defender options.
- `*_response_hash`: SHA-256 hashes for traceability without raw response text.
- `*_temperature`, `*_top_p`, `*_max_tokens`, `*_seed`: generation settings.
- `parse_failure_flag`, `all_parse_valid`, `task_consistency_label`: parsing and quality diagnostics.

See `results/canonical_run/analysis_rows_public_column_policy.json` for the exact public column policy.
