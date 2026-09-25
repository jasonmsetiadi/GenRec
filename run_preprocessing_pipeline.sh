#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS=(
Musical_Instruments
Beauty
Baby
Toys_and_Games
)
DATA_VERSION="${DATA_VERSION:-14}"
LOCAL_TEXT_MODEL="${LOCAL_TEXT_MODEL:-sentence-transformers/sentence-t5-base}"
PCA_DIM="${PCA_DIM:-512}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"

if (( $# > 0 )); then
  DATASETS=("$@")
fi

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

for dataset in "${DATASETS[@]}"; do
  echo "[$dataset] Step 1/4: Downloading data."
  (
    cd "$ROOT_DIR/preprocessing"
    "$PYTHON_BIN" download_data.py \
      --source amazon \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION"
  )

  echo "[$dataset] Step 2/4: Preprocessing data."
  (
    cd "$ROOT_DIR/preprocessing"
    "$PYTHON_BIN" process_data.py \
      --dataset_type amazon \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION"
  )

  echo "[$dataset] Step 3/4: Generating local $LOCAL_TEXT_MODEL text embeddings."
  (
    cd "$ROOT_DIR/preprocessing"
    "$PYTHON_BIN" process_embedding.py \
      --embedding_type text_local \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --model_name_or_path "$LOCAL_TEXT_MODEL" \
      --pca_dim "$PCA_DIM" \
      --batch_size "$BATCH_SIZE" \
      --gpu_id "$GPU_ID"
  )

  echo "[$dataset] Step 4/4: Quantizing text embeddings."
  (
    cd "$ROOT_DIR/quantization"
    "$PYTHON_BIN" main.py \
      --model_name rqvae \
      --dataset_name "$dataset" \
      --embedding_modality text \
      --embedding_model "$LOCAL_TEXT_MODEL" \
      --data_base_path "$ROOT_DIR/datasets" \
      --codebook_base_path "$ROOT_DIR/datasets"
  )

  echo "[$dataset] Completed. Codebook: $ROOT_DIR/datasets/$dataset/codebooks/"
done
