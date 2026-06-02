# Prompt Templates

These prompt templates implement the three-stage experiment flow:

1. `attacker.yaml`
2. `defender.yaml`
3. `evaluator.yaml`
4. `runtime_input_format.yaml`
5. `branches/baseline.yaml`
6. `branches/control.yaml`
7. `branches/deception.yaml`

All templates assume each scenario JSON provides:

- `scenario_id`
- `topic`
- `context_document`
- `deceptive_goal`
- `decision_options`
- `target_option.index`
- `target_option.text`

The attacker sees the target option. The defender does not. The evaluator sees both.

## Runtime Layout

- `runtime_input_format.yaml`
  - Canonical field contract for one scenario-level run record
- `branches/baseline.yaml`
  - No attacker stage; defender decides directly from context
- `branches/control.yaml`
  - Attacker argues for the target option fairly, without deception
- `branches/deception.yaml`
  - Attacker uses the deceptive persuasion objective

Each branch template binds the shared scenario fields into a concrete stage sequence that can be executed by a runner.
