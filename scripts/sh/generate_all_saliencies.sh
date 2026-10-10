#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/py/generate_saliencies.py"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_ROOT="${DATA_ROOT}/saliencies"
TARGET_MODE="true"
METHODS=(gradcam guided_gradcam gradcampp xgradcam rise lime)

if [[ ! -x "${PYTHON}" ]]; then
  echo "Virtual environment python not found at ${PYTHON}" >&2
  exit 1
fi

run_saliencies() {
  local dataset="$1"
  local model="$2"
  local checkpoint="$3"
  local stem=""
  local output_dir=""

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping missing checkpoint: ${checkpoint}"
    return
  fi

  stem="$(basename "${checkpoint}" .pt)"
  for method in "${METHODS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${dataset}/${stem}/${model}/${method}/${TARGET_MODE}"
    if [[ -d "${output_dir}" ]]; then
      echo "Skipping existing saliency maps: ${output_dir}"
      continue
    fi

    "${PYTHON}" "${SCRIPT}" \
      "${dataset}" \
      "${method}" \
      "${checkpoint}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --target-mode "${TARGET_MODE}"
  done
}

# Oxford Pet / ResNet50
run_saliencies oxford_pet resnet50 "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_0.pt"
run_saliencies oxford_pet resnet50 "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_1.pt"
run_saliencies oxford_pet resnet50 "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_2.pt"
run_saliencies oxford_pet resnet50 "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_3.pt"
run_saliencies oxford_pet resnet50 "${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_4.pt"

# Oxford Pet / ViT-B-16
# run_saliencies oxford_pet vit_b_16 "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_0.pt"
# run_saliencies oxford_pet vit_b_16 "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_1.pt"
# run_saliencies oxford_pet vit_b_16 "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_2.pt"
# run_saliencies oxford_pet vit_b_16 "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_3.pt"
# run_saliencies oxford_pet vit_b_16 "${CHECKPOINT_DIR}/oxford_pet_vit_b_16_pretrained_seed_4.pt"

# # Caltech101 / ResNet50
# run_saliencies caltech101 resnet50 "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_0.pt"
# run_saliencies caltech101 resnet50 "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_1.pt"
# run_saliencies caltech101 resnet50 "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_2.pt"
# run_saliencies caltech101 resnet50 "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_3.pt"
# run_saliencies caltech101 resnet50 "${CHECKPOINT_DIR}/caltech101_resnet50_pretrained_seed_4.pt"

# # Caltech101 / ViT-B-16
# run_saliencies caltech101 vit_b_16 "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_0.pt"
# run_saliencies caltech101 vit_b_16 "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_1.pt"
# run_saliencies caltech101 vit_b_16 "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_2.pt"
# run_saliencies caltech101 vit_b_16 "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_3.pt"
# run_saliencies caltech101 vit_b_16 "${CHECKPOINT_DIR}/caltech101_vit_b_16_pretrained_seed_4.pt"

# # FGVC Aircraft / ResNet50
# run_saliencies fgvc_aircraft resnet50 "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_0.pt"
# run_saliencies fgvc_aircraft resnet50 "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_1.pt"
# run_saliencies fgvc_aircraft resnet50 "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_2.pt"
# run_saliencies fgvc_aircraft resnet50 "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_3.pt"
# run_saliencies fgvc_aircraft resnet50 "${CHECKPOINT_DIR}/fgvc_aircraft_resnet50_pretrained_seed_4.pt"

# # FGVC Aircraft / ViT-B-16
# run_saliencies fgvc_aircraft vit_b_16 "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_0.pt"
# run_saliencies fgvc_aircraft vit_b_16 "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_1.pt"
# run_saliencies fgvc_aircraft vit_b_16 "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_2.pt"
# run_saliencies fgvc_aircraft vit_b_16 "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_3.pt"
# run_saliencies fgvc_aircraft vit_b_16 "${CHECKPOINT_DIR}/fgvc_aircraft_vit_b_16_pretrained_seed_4.pt"

# # Stanford Cars / ResNet50
# run_saliencies stanford_cars resnet50 "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_0.pt"
# run_saliencies stanford_cars resnet50 "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_1.pt"
# run_saliencies stanford_cars resnet50 "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_2.pt"
# run_saliencies stanford_cars resnet50 "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_3.pt"
# run_saliencies stanford_cars resnet50 "${CHECKPOINT_DIR}/stanford_cars_resnet50_pretrained_seed_4.pt"

# # Stanford Cars / ViT-B-16
# run_saliencies stanford_cars vit_b_16 "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_0.pt"
# run_saliencies stanford_cars vit_b_16 "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_1.pt"
# run_saliencies stanford_cars vit_b_16 "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_2.pt"
# run_saliencies stanford_cars vit_b_16 "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_3.pt"
# run_saliencies stanford_cars vit_b_16 "${CHECKPOINT_DIR}/stanford_cars_vit_b_16_pretrained_seed_4.pt"
