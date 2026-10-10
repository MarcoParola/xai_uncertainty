#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_dataset.py"
DATA_ROOT="${PROJECT_ROOT}/data"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Virtual environment python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" "${DOWNLOAD_SCRIPT}" oxford_pet --root "${DATA_ROOT}" --split all
"${PYTHON}" "${DOWNLOAD_SCRIPT}" caltech101 --root "${DATA_ROOT}" --split all
"${PYTHON}" "${DOWNLOAD_SCRIPT}" fgvc_aircraft --root "${DATA_ROOT}" --split all
"${PYTHON}" "${DOWNLOAD_SCRIPT}" stanford_cars --root "${DATA_ROOT}" --split all
