#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workers="${1:-3}"
run_dir="${2:-${RERUN_ATTACKER_NONDECEPTIVE_RUN_DIR:-}}"
if ! [[ "${workers}" =~ ^[0-9]+$ ]] || [[ "${workers}" -lt 1 ]]; then
  echo "Usage: $0 [worker_count] [run_dir]" >&2
  exit 1
fi
if [[ -z "${run_dir}" ]]; then
  echo "Missing run_dir. Pass an existing outputs/runs/... directory as arg 2." >&2
  exit 1
fi

if [[ ! -f .env.local ]]; then
  echo "Missing .env.local. Create it from .env.example before running." >&2
  exit 1
fi

mkdir -p logs
set -a
source .env.local
set +a

timestamp="$(date +%Y%m%d_%H%M%S)"
pid_manifest="logs/rerun_attacker_nondeceptive_${timestamp}.pids"
latest_pid_manifest="logs/rerun_attacker_nondeceptive.pids"
recover_stale_minutes="${RECOVER_STALE_MINUTES:-180}"
extra_args="${RERUN_ATTACKER_NONDECEPTIVE_EXTRA_ARGS:---incomplete-only --recover-stale-minutes ${recover_stale_minutes}}"
: > "${pid_manifest}"

for shard_index in $(seq 0 $((workers - 1))); do
  log_path="logs/rerun_attacker_nondeceptive_${timestamp}_worker_${shard_index}_of_${workers}.log"
  worker_script="logs/rerun_attacker_nondeceptive_${timestamp}_worker_${shard_index}_of_${workers}.sh"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' 'cd "$(dirname "$0")/.."'
    printf '%s\n' 'set -a'
    printf '%s\n' 'source .env.local'
    printf '%s\n' 'set +a'
    printf '%s\n' 'py="${PYTHON_BIN:-python3}"'
    printf '%s\n' '[[ -x .venv/bin/python ]] && py=".venv/bin/python"'
    printf '%s\n' 'export PYTHONUNBUFFERED=1'
    printf '%s\n' "echo \"START worker=${shard_index}/${workers} \$(date '+%Y-%m-%dT%H:%M:%S%z') run_dir=${run_dir}\""
    printf '%s\n' 'set +e'
    printf '%s\n' "\"\${py}\" scripts/rerun_attacker_nondeceptive.py --output-dir '${run_dir}' --num-shards ${workers} --shard-index ${shard_index} ${extra_args}"
    printf '%s\n' 'exit_code="$?"'
    printf '%s\n' 'set -e'
    printf '%s\n' "echo \"END worker=${shard_index}/${workers} exit_code=\${exit_code} \$(date '+%Y-%m-%dT%H:%M:%S%z')\""
    printf '%s\n' 'exit "${exit_code}"'
  } > "${worker_script}"
  chmod +x "${worker_script}"
  nohup bash "${worker_script}" > "${log_path}" 2>&1 &
  pid="$!"
  echo "${pid} ${shard_index} ${log_path} ${worker_script}" >> "${pid_manifest}"
  echo "worker=${shard_index}/${workers} pid=${pid} log=${log_path} script=${worker_script}"
done

cp "${pid_manifest}" "${latest_pid_manifest}"
echo "PID_MANIFEST=${pid_manifest}"
echo "After all workers finish, run:"
echo "python3 scripts/merge_run_outputs.py --run-dir ${run_dir} --strict"
echo "python3 scripts/check_run_progress.py --run-dir ${run_dir}"
