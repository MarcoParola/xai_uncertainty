#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/env/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/py/generate_saliencies.py"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_ROOT="${DATA_ROOT}/saliencies"
TARGET_MODE="true"
NUM_IMAGES=600
SAMPLE_SEED=42
DEVICE="cuda"
BATCH_SIZE=16
RISE_IMAGE_BATCH_SIZE=1
RISE_MASK_BATCH_SIZE=32
RISE_MASKS=2000
METHODS=(gradcam gradcampp lime rise)
SEEDS=(0 1 2 3)
LABELS=(best minus_03 minus_05 minus_07 minus_10 minus_15 plus_03 plus_05 plus_07 plus_10 plus_15)
TOTAL_RUNS=$((${#SEEDS[@]} * ${#LABELS[@]} * ${#METHODS[@]}))
RUNS_DONE=0
GENERATED_RUNS=0
REUSED_RUNS=0
MISSING_RUNS=0

if [[ ! -x "${PYTHON}" ]]; then
  echo "Virtual environment python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c 'import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Saliency generation requires a GPU; no CPU fallback will be used.")
torch.ones(1, device="cuda")
print("GPU confirmed:", torch.cuda.get_device_name(0), flush=True)'

show_progress() {
  local width=30
  local filled=$((RUNS_DONE * width / TOTAL_RUNS))
  local bar=""
  local remaining=""
  printf -v bar '%*s' "${filled}" ''
  printf -v remaining '%*s' "$((width - filled))" ''
  printf '[%s%s] %3d%% %d/%d runs processed (%d generated, %d reused, %d missing)\n' \
    "${bar// /#}" "${remaining// /-}" "$((RUNS_DONE * 100 / TOTAL_RUNS))" \
    "${RUNS_DONE}" "${TOTAL_RUNS}" "${GENERATED_RUNS}" "${REUSED_RUNS}" "${MISSING_RUNS}" >&2
}

run_saliencies() {
  local dataset="$1"
  local model="$2"
  local checkpoint="$3"
  local stem=""
  local output_dir=""
  local batch_size=""
  local extra_args=()

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping missing checkpoint: ${checkpoint}"
    RUNS_DONE=$((RUNS_DONE + ${#METHODS[@]}))
    MISSING_RUNS=$((MISSING_RUNS + ${#METHODS[@]}))
    show_progress
    return
  fi

  stem="$(basename "${checkpoint}" .pt)"
  for method in "${METHODS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${dataset}/${stem}/${model}/${method}/${TARGET_MODE}"
    if [[ -f "${output_dir}/manifest.json" ]]; then
      if [[ "${method}" != "rise" ]] || "${PYTHON}" -c 'import json, sys
with open(sys.argv[1]) as handle:
    manifest = json.load(handle)
sys.exit(0 if manifest.get("rise_masks") == int(sys.argv[2]) else 1)' "${output_dir}/manifest.json" "${RISE_MASKS}"; then
        echo "Skipping existing saliency maps: ${output_dir}"
        RUNS_DONE=$((RUNS_DONE + 1))
        REUSED_RUNS=$((REUSED_RUNS + 1))
        show_progress
        continue
      fi
    fi
    batch_size="${BATCH_SIZE}"
    extra_args=()
    if [[ "${method}" == "rise" ]]; then
      batch_size="${RISE_IMAGE_BATCH_SIZE}"
      extra_args+=(--rise-batch-size "${RISE_MASK_BATCH_SIZE}" --rise-masks "${RISE_MASKS}")
    fi
    if [[ -d "${output_dir}" ]]; then
      if [[ -f "${output_dir}/manifest.json" ]]; then
        echo "Restarting RISE output with ${RISE_MASKS} masks (previous count differs or is unknown): ${output_dir}"
      else
        echo "Restarting incomplete saliency output: ${output_dir}"
      fi
      extra_args+=(--overwrite)
    fi

    echo "Generating run $((RUNS_DONE + 1))/${TOTAL_RUNS}: ${stem} / ${method}"
    "${PYTHON}" "${SCRIPT}" \
      "${dataset}" \
      "${method}" \
      "${checkpoint}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --num-images "${NUM_IMAGES}" \
      --batch-size "${batch_size}" \
      --seed "${SAMPLE_SEED}" \
      --target-mode "${TARGET_MODE}" \
      --device "${DEVICE}" \
      "${extra_args[@]}"
    RUNS_DONE=$((RUNS_DONE + 1))
    GENERATED_RUNS=$((GENERATED_RUNS + 1))
    show_progress
  done
}

# Oxford Pet / ResNet50
show_progress
for seed in "${SEEDS[@]}"; do
  stem="${CHECKPOINT_DIR}/oxford_pet_resnet50_pretrained_seed_${seed}"
  # The unsuffixed checkpoint duplicates _best; generate it only once.
  for label in "${LABELS[@]}"; do
    run_saliencies oxford_pet resnet50 "${stem}_${label}.pt"
  done
done

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
