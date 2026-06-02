# Deception Benchmark V2

Public release package for a refusal-censored target-aware LLM evaluation benchmark.

Generated: 2026-06-02T23:24:12+00:00

## What Is Included

- `deception_benchmark/`: benchmark runtime, model clients, parsers, gate assignment, and derived outcome logic.
- `scripts/`: experiment orchestration, merge, summary, validation, and figure/reanalysis scripts.
- `prompts/`: attacker, defender, evaluator, and branch prompt templates.
- `data/normalized-scenarios/`: 53 normalized scenario JSON files.
- `data/derived/`: scenario registry, joined design rows, branch/run tables, and scenario summaries.
- `data/manual/`: model covariates, scenario metadata, and complexity annotations.
- `results/canonical_run/analysis_rows_public.*`: canonical 2,968-row evidence lock with raw text columns removed.
- `results/final_results_20260427/`: aggregate result tables and diagnostics.
- `validation/human_validation_v2/`: active V2 validation summaries and adjudicated tactic audit artifacts.
- `figures/` and `paper/`: paper figures and LaTeX/PDF source snapshot.

## What Is Excluded From Public GitHub

- API keys, `.env.local`, local app settings, virtual environments, caches, logs, and `.DS_Store`.
- Raw provider response directories, checkpoint databases, and judge logs.
- Raw generated attacker/defender/gate text in the canonical analysis table.
- Full human-validation annotation packets that contain raw generated text.
- Superseded archive runs and old validation attempts.

The public analysis table keeps hashes, parsed outcomes, gate assignments, model families, covariates, and diagnostics. Scenario definitions are released separately in `data/`.

## Headline Counts

- Canonical rows: 2,968.
- Generated target-aware interventions: 506/2,968 = 17.0%.
- Generated-subset conditional target shift: 368/506 = 72.7%.
- Full-denominator realized exposure: 368/2,968 = 12.4%.
- Policy-gate observed exposure: 0/1,013 = 0.0%.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_run_table.py
python scripts/verify_public_release.py
```

Expected design counts:

- scenarios: 53
- joined rows: 2,968
- branch interactions: 8,904

To run a dry-run smoke test without API calls:

```bash
python scripts/run_experiment.py --dry-run --limit 3 --run-id dry_run_demo --consolidate
```

Real API-backed runs require `.env.local` created from `.env.example`.

## Citation

Use `CITATION.cff` once author metadata is finalized.

## Release Boundary

This repository is for measurement and reproducibility. It should not be used as a library of manipulative prompts or generated interventions. See `RESPONSIBLE_RELEASE.md`.
