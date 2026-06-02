# Reproduction

## Verify the public package

```bash
python scripts/verify_public_release.py
```

## Rebuild design tables

```bash
python scripts/generate_run_table.py
```

## Recompute summary tables from public analysis rows

```bash
python scripts/build_summary_tables.py \
  --analysis-rows results/canonical_run/analysis_rows_public.jsonl \
  --scenario-registry data/derived/scenario_registry.csv \
  --output-dir outputs/tables_public
```

## Run a dry-run benchmark smoke test

```bash
python scripts/run_experiment.py --dry-run --limit 3 --run-id dry_run_demo --consolidate
```

## Run API-backed benchmark jobs

```bash
cp .env.example .env.local
# edit .env.local with your keys
./scripts/run_sharded_nohup.sh 4 my_new_run
python scripts/merge_run_outputs.py --run-dir outputs/runs/my_new_run --strict
```
