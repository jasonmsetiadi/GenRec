#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="$ROOT_DIR/datasets"
DATASET="${DATASET:-Baby}"
MODEL="${MODEL:-TIGER}"
QUANT_METHOD="${QUANT_METHOD:-rqvae}"
EMBEDDING_MODALITY="${EMBEDDING_MODALITY:-text}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/sentence-t5-base}"
SID_LEVELS="${SID_LEVELS:-5}"
TRUNCATION_SEED="${TRUNCATION_SEED:-2023}"

CODEBOOK_DIR="$DATA_DIR/$DATASET/codebooks"
FIXED_CODEBOOK="$CODEBOOK_DIR/$DATASET.$EMBEDDING_MODALITY.$QUANT_METHOD.npy"
VARIABLE_CODEBOOK="$CODEBOOK_DIR/$DATASET.$EMBEDDING_MODALITY.$QUANT_METHOD.varlen.npy"
FIXED_RUN="fixed-$QUANT_METHOD"
VARIABLE_RUN="varlen-$QUANT_METHOD"
RUN_DIR="$ROOT_DIR/ckpt/recommendation/$DATASET/${MODEL}_${QUANT_METHOD}"

format_duration() {
  local total_seconds="$1"
  printf '%02dh %02dm %02ds' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

PIPELINE_START="$SECONDS"

echo "Prerequisite: $DATA_DIR/$DATASET must contain the preprocessed interactions and embeddings."
echo
echo "Step 1/4: Generate the fixed-length SID codebook."
STEP_START="$SECONDS"
(
  cd "$ROOT_DIR/quantization"
  "$PYTHON_BIN" main.py \
    --model_name "$QUANT_METHOD" \
    --dataset_name "$DATASET" \
    --embedding_modality "$EMBEDDING_MODALITY" \
    --embedding_model "$EMBEDDING_MODEL" \
    --data_base_path "$DATA_DIR" \
    --codebook_base_path "$DATA_DIR"
)
echo "Step 1/4 completed in $(format_duration "$((SECONDS - STEP_START))")."

echo
echo "Step 2/4: Train the fixed-length baseline."
STEP_START="$SECONDS"
(
  cd "$ROOT_DIR/recommendation"
  "$PYTHON_BIN" main.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --quant_method "$QUANT_METHOD" \
    --embedding_modality "$EMBEDDING_MODALITY" \
    --run_name "$FIXED_RUN"
)
echo "Step 2/4 completed in $(format_duration "$((SECONDS - STEP_START))")."

echo
echo "Step 3/4: Create a reproducible variable-length test codebook from the fixed codebook."
STEP_START="$SECONDS"
"$PYTHON_BIN" "$ROOT_DIR/quantization/truncate_codebook.py" \
  --input "$FIXED_CODEBOOK" \
  --output "$VARIABLE_CODEBOOK" \
  --min-length 1 \
  --max-length "$SID_LEVELS" \
  --seed "$TRUNCATION_SEED"
echo "Step 3/4 completed in $(format_duration "$((SECONDS - STEP_START))")."

echo
echo "Step 4/4: Train with variable-length SIDs."
STEP_START="$SECONDS"
(
  cd "$ROOT_DIR/recommendation"
  "$PYTHON_BIN" main.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --quant_method "$QUANT_METHOD" \
    --embedding_modality "$EMBEDDING_MODALITY" \
    --codebook_path "$VARIABLE_CODEBOOK" \
    --no_dedup_layer \
    --sid_num_levels "$SID_LEVELS" \
    --run_name "$VARIABLE_RUN"
)
echo "Step 4/4 completed in $(format_duration "$((SECONDS - STEP_START))")."

echo
echo "Comparison complete in $(format_duration "$((SECONDS - PIPELINE_START))")."
echo "Fixed-length log:    $RUN_DIR/$FIXED_RUN/training.log"
echo "Variable-length log: $RUN_DIR/$VARIABLE_RUN/training.log"
echo "Compare Recall@K and NDCG@K with identical dataset, model, and training settings."
