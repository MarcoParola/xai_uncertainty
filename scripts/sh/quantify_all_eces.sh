#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/py/quantify_uncertainty.py"
DATA_ROOT="${PROJECT_ROOT}/data"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
RESULTS_DIR="${PROJECT_ROOT}/results/uncertainty"
ECE_BINS="${ECE_BINS:-15}"
DEVICE="${DEVICE:-}"
device_args=()

if [[ -n "${DEVICE}" ]]; then
  device_args=(--device "${DEVICE}")
fi

mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Virtual environment python not found at ${PYTHON}" >&2
  exit 1
fi

checkpoints=(
  "${CHECKPOINT_DIR}"/*_best.pt
  "${CHECKPOINT_DIR}"/*_minus_*.pt
  "${CHECKPOINT_DIR}"/*_plus_*.pt
)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No epoch-specific checkpoints found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  case "${checkpoint_name}" in
    oxford_pet_*) dataset="oxford_pet"; rest="${checkpoint_name#oxford_pet_}" ;;
    caltech101_*) dataset="caltech101"; rest="${checkpoint_name#caltech101_}" ;;
    fgvc_aircraft_*) dataset="fgvc_aircraft"; rest="${checkpoint_name#fgvc_aircraft_}" ;;
    stanford_cars_*) dataset="stanford_cars"; rest="${checkpoint_name#stanford_cars_}" ;;
    imagenette_*) dataset="imagenette"; rest="${checkpoint_name#imagenette_}" ;;
    *) echo "Skipping unrecognized checkpoint: ${checkpoint}"; continue ;;
  esac

  case "${rest}" in
    resnet50_*) model="resnet50" ;;
    vit_b_16_*) model="vit_b_16" ;;
    *) echo "Skipping unrecognized model checkpoint: ${checkpoint}"; continue ;;
  esac

  csv_path="${RESULTS_DIR}/${dataset}_${model}_ece.csv"
  echo "Computing ECE ${checkpoint} -> ${csv_path}"
  "${PYTHON}" "${SCRIPT}" \
    --checkpoint "${checkpoint}" \
    --data-root "${DATA_ROOT}" \
    --metrics ece \
    --ece-bins "${ECE_BINS}" \
    --csv "${csv_path}" \
    "${device_args[@]}"
done
