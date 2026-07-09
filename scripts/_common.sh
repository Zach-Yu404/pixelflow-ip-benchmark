#!/usr/bin/env bash
# Shared setup for all scripts. Sourced, not run. Establishes project root, the
# python interpreter (from configs/paths.local.yaml), and the command log.
set -euo pipefail

# Resolve project root = parent of scripts/ regardless of caller CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PATHS_YAML="${PROJECT_ROOT}/configs/paths.local.yaml"
PY="$(grep -E '^python_bin:' "${PATHS_YAML}" | awk '{print $2}')"
PY="${PY/#\~/$HOME}"   # expand a leading ~ (paths.local.yaml keeps it user-neutral)
[ -x "${PY}" ] || PY="python"
DEFAULT_GPU="$(grep -E '^default_gpu:' "${PATHS_YAML}" | awk '{print $2}')"
: "${GPU:=${DEFAULT_GPU:-0}}"

LOG_DIR="${PROJECT_ROOT}/docs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/reproduction_commands.log"

log_cmd() {  # echo + append the exact command being run, with a timestamp
  local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "${LOG}"
}

export PROJECT_ROOT PY PATHS_YAML GPU LOG
