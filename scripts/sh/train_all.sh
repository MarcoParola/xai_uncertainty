#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/py/training_classifier.py"
DATA_ROOT="${PROJECT_ROOT}/data"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
export TORCH_HOME="${PROJECT_ROOT}/.torch"

mkdir -p "${CHECKPOINT_DIR}" "${TORCH_HOME}"

run_training() {
  local output=""
  local best_output=""
  local args=("$@")

  for ((index = 0; index < ${#args[@]}; index++)); do
    if [[ "${args[index]}" == "--output" && $((index + 1)) -lt ${#args[@]} ]]; then
      output="${args[index + 1]}"
      break
    fi
  done

  if [[ -n "${output}" ]]; then
    best_output="${output%.pt}_best.pt"
  fi

  if [[ -f "${output}" && -f "${best_output}" ]]; then
    echo "Skipping completed training: ${output}"
    return
  fi

  "${PYTHON}" "${TRAIN_SCRIPT}" "$@"
}

# Oxford Pet / ResNet50
run_training --dataset oxford_pet --model resnet50 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_0.pt"
run_training --dataset oxford_pet --model resnet50 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_1.pt"
run_training --dataset oxford_pet --model resnet50 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_2.pt"
run_training --dataset oxford_pet --model resnet50 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_3.pt"
run_training --dataset oxford_pet --model resnet50 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_4.pt"

# Oxford Pet / ViT-B-16
run_training --dataset oxford_pet --model vit_b_16 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_0.pt"
run_training --dataset oxford_pet --model vit_b_16 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_1.pt"
run_training --dataset oxford_pet --model vit_b_16 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_2.pt"
run_training --dataset oxford_pet --model vit_b_16 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_3.pt"
run_training --dataset oxford_pet --model vit_b_16 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_4.pt"

# Caltech101 / ResNet50
run_training --dataset caltech101 --model resnet50 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_0.pt"
run_training --dataset caltech101 --model resnet50 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_1.pt"
run_training --dataset caltech101 --model resnet50 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_2.pt"
run_training --dataset caltech101 --model resnet50 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_3.pt"
run_training --dataset caltech101 --model resnet50 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_4.pt"

# Caltech101 / ViT-B-16
run_training --dataset caltech101 --model vit_b_16 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_0.pt"
run_training --dataset caltech101 --model vit_b_16 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_1.pt"
run_training --dataset caltech101 --model vit_b_16 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_2.pt"
run_training --dataset caltech101 --model vit_b_16 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_3.pt"
run_training --dataset caltech101 --model vit_b_16 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_4.pt"

# FGVC Aircraft / ResNet50
run_training --dataset fgvc_aircraft --model resnet50 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_0.pt"
run_training --dataset fgvc_aircraft --model resnet50 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_1.pt"
run_training --dataset fgvc_aircraft --model resnet50 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_2.pt"
run_training --dataset fgvc_aircraft --model resnet50 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_3.pt"
run_training --dataset fgvc_aircraft --model resnet50 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_4.pt"

# FGVC Aircraft / ViT-B-16
run_training --dataset fgvc_aircraft --model vit_b_16 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_0.pt"
run_training --dataset fgvc_aircraft --model vit_b_16 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_1.pt"
run_training --dataset fgvc_aircraft --model vit_b_16 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_2.pt"
run_training --dataset fgvc_aircraft --model vit_b_16 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_3.pt"
run_training --dataset fgvc_aircraft --model vit_b_16 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_4.pt"

# Stanford Cars / ResNet50
run_training --dataset stanford_cars --model resnet50 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_0.pt"
run_training --dataset stanford_cars --model resnet50 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_1.pt"
run_training --dataset stanford_cars --model resnet50 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_2.pt"
run_training --dataset stanford_cars --model resnet50 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_3.pt"
run_training --dataset stanford_cars --model resnet50 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_4.pt"

# Stanford Cars / ViT-B-16
run_training --dataset stanford_cars --model vit_b_16 --seed 0 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_0.pt"
run_training --dataset stanford_cars --model vit_b_16 --seed 1 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_1.pt"
run_training --dataset stanford_cars --model vit_b_16 --seed 2 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_2.pt"
run_training --dataset stanford_cars --model vit_b_16 --seed 3 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_3.pt"
run_training --dataset stanford_cars --model vit_b_16 --seed 4 --data-root "${DATA_ROOT}" --output "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_4.pt"
