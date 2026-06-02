#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env.local ]]; then
  echo "Missing .env.local. Create it from .env.example before running." >&2
  exit 1
fi

mkdir -p logs
set -a
source .env.local
set +a

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${1:-tacl_revision_gate_iv_${timestamp}}"
log_path="logs/full_experiment_${timestamp}.log"
pid_path="logs/full_experiment_${timestamp}.pid"
latest_pid_path="logs/full_experiment.pid"
worker_script="logs/full_experiment_${timestamp}.sh"
extra_args="${RUN_EXPERIMENT_EXTRA_ARGS:-}"
quoted_run_id="$(printf '%q' "${run_id}")"

{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'set -euo pipefail'
  printf '%s\n' 'cd "$(dirname "$0")/.."'
  printf '%s\n' 'set -a'
  printf '%s\n' 'source .env.local'
  printf '%s\n' 'set +a'
  printf '%s\n' 'py="${PYTHON_BIN:-python3}"'
  printf '%s\n' '[[ -x .venv/bin/python ]] && py=".venv/bin/python"'
  printf '%s\n' "run_id=${quoted_run_id}"
  printf '%s\n' 'echo "START full_experiment run_id=${run_id} $(date '\''+%Y-%m-%dT%H:%M:%S%z'\'')"'
  printf '%s\n' "\"\${py}\" scripts/run_experiment.py --run-id \"\${run_id}\" --iv-design gate_only --resume --recover-stale-minutes \"\${RECOVER_STALE_MINUTES:-180}\" --consolidate ${extra_args}"
  printf '%s\n' '"${py}" scripts/export_annotation_subsets.py --analysis-rows "outputs/runs/${run_id}/analysis_rows.jsonl" --output-dir "outputs/runs/${run_id}/subsets"'
  printf '%s\n' '"${py}" scripts/build_summary_tables.py --analysis-rows "outputs/runs/${run_id}/analysis_rows.jsonl" --subset-manifest "outputs/runs/${run_id}/subsets/subset_manifest.json" --output-dir "outputs/runs/${run_id}/tables"'
  printf '%s\n' 'exit_code="$?"'
  printf '%s\n' 'echo "END full_experiment exit_code=${exit_code} $(date '\''+%Y-%m-%dT%H:%M:%S%z'\'')"'
  printf '%s\n' 'exit "${exit_code}"'
} > "${worker_script}"
chmod +x "${worker_script}"

nohup bash "${worker_script}" > "${log_path}" 2>&1 &
pid="$!"
echo "${pid}" > "${pid_path}"
echo "${pid}" > "${latest_pid_path}"
echo "PID=${pid}"
echo "LOG=${log_path}"
echo "PID_FILE=${pid_path}"
echo "SCRIPT=${worker_script}"
echo "RUN_ID=${run_id}"
