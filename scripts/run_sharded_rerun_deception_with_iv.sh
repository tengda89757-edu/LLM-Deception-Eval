#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workers="${1:-4}"
if ! [[ "${workers}" =~ ^[0-9]+$ ]] || [[ "${workers}" -lt 1 ]]; then
  echo "Usage: $0 [worker_count]" >&2
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
pid_manifest="logs/rerun_deception_with_iv_${timestamp}.pids"
latest_pid_manifest="logs/rerun_deception_with_iv.pids"
extra_args="${RERUN_DECEPTION_EXTRA_ARGS:-}"
: > "${pid_manifest}"

for shard_index in $(seq 0 $((workers - 1))); do
  log_path="logs/rerun_deception_with_iv_${timestamp}_worker_${shard_index}_of_${workers}.log"
  worker_script="logs/rerun_deception_with_iv_${timestamp}_worker_${shard_index}_of_${workers}.sh"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' 'cd "$(dirname "$0")/.."'
    printf '%s\n' 'set -a'
    printf '%s\n' 'source .env.local'
    printf '%s\n' 'set +a'
    printf '%s\n' 'py="${PYTHON_BIN:-python3}"'
    printf '%s\n' '[[ -x .venv/bin/python ]] && py=".venv/bin/python"'
    printf '%s\n' "echo \"START worker=${shard_index}/${workers} \$(date '+%Y-%m-%dT%H:%M:%S%z')\""
    printf '%s\n' "\"\${py}\" scripts/rerun_deception_with_iv.py --num-shards ${workers} --shard-index ${shard_index} ${extra_args}"
    printf '%s\n' 'exit_code="$?"'
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
echo "  python revision_supplementary/scripts/heckman_reanalysis.py"
