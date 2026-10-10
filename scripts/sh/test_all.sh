#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
TEST_SCRIPT="${PROJECT_ROOT}/scripts/py/test_classifier.py"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
RESULTS_DIR="${PROJECT_ROOT}/results"

mkdir -p "${RESULTS_DIR}"
rm -f "${RESULTS_DIR}"/*.csv

csv_for_checkpoint() {
  local checkpoint_name="$1"
  local dataset="unknown"
  local model="unknown"
  local rest=""

  case "${checkpoint_name}" in
    oxford_pet_*) dataset="oxford_pet"; rest="${checkpoint_name#oxford_pet_}" ;;
    caltech101_*) dataset="caltech101"; rest="${checkpoint_name#caltech101_}" ;;
    fgvc_aircraft_*) dataset="fgvc_aircraft"; rest="${checkpoint_name#fgvc_aircraft_}" ;;
    stanford_cars_*) dataset="stanford_cars"; rest="${checkpoint_name#stanford_cars_}" ;;
    *) rest="${checkpoint_name}" ;;
  esac

  case "${rest}" in
    resnet50_*) model="resnet50" ;;
    vit_b_16_*) model="vit_b_16" ;;
  esac

  echo "${RESULTS_DIR}/${dataset}_${model}.csv"
}

checkpoints=("${CHECKPOINT_DIR}"/*.pt)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No checkpoints found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  csv_path="$(csv_for_checkpoint "${checkpoint_name}")"
  echo "Testing ${checkpoint} -> ${csv_path}"
  "${PYTHON}" "${TEST_SCRIPT}" --checkpoint "${checkpoint}" --csv "${csv_path}"
done
