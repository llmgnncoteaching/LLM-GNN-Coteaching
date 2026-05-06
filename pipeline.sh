#!/bin/bash
set -e  # Stop on first error

# ===== CONFIGURATION =====
if [ -f "config.sh" ]; then
    source config.sh
fi

# Command line arguments
DATASET=${1:-"cora"}
SHOT_COUNT=${2:-"5"}
SEED=${3:-"42"}
NUM_ROUNDS=${4:-"3"}
BASE_MODEL_PATH=${5:-"${BASE_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"}

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
LF_DIR="${LF_DIR:-$PROJECT_DIR/LLaMA-Factory}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspace}"

# ===== ENVIRONMENT SETUP =====
# Respect CUDA_VISIBLE_DEVICES if already set; otherwise auto-detect (skip ERR GPUs)
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | grep -v '\[N/A\]\|ERR' | cut -d',' -f1 | tr -d ' ' | paste -sd,)
  if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="0"
  fi
  echo "Auto-detected usable GPUs: $CUDA_VISIBLE_DEVICES"
else
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Number of GPUs: $NUM_GPUS"
# GNN always uses first visible GPU (cuda:0 inside the process)
DEVICE="cuda:0"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export DISABLE_VERSION_CHECK=1

MAIN_EXP_DIR="$WORKSPACE_DIR/results/co_teaching"
DATASET_DIR="$LF_DIR/data"
DATASET_INFO_FILE="$DATASET_DIR/dataset_info.json"
mkdir -p "$MAIN_EXP_DIR" "$DATASET_DIR"

# Conda (skip if already in correct env)
CONDA_SH="${CONDA_SH:-$(find /home /opt -maxdepth 5 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)}"
if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "${ENV_NAME:-data_env}" 2>/dev/null || true
fi

# ===== PIPELINE EXECUTION =====
RUN_ID="${DATASET}_${SHOT_COUNT}shot_seed${SEED}${EXP_TAG:+_$EXP_TAG}"
RUN_DIR="$MAIN_EXP_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

SFT_DATASET_PREFIX="${DATASET}_sft_${SHOT_COUNT}_shot"
GNN_MODEL_PATH="$RUN_DIR/gnn_round0.pt"

echo "=== Starting LLM-GNN Co-Teaching Pipeline for $RUN_ID ==="
echo "=== Rounds: $NUM_ROUNDS ==="
echo "=== Model: $BASE_MODEL_PATH ==="
echo "=== Project: $PROJECT_DIR ==="


# ===========================================================================
# Helper functions (defined early so all stages can use them)
# ===========================================================================
run_llm_inference() {
  local ADAPTER="$1"
  local DATASET_KEY="$2"
  local OUTPUT_FILE="$3"
  
  cd "$PROJECT_DIR"
  VLLM_LOGGING_LEVEL=ERROR VLLM_CONFIGURE_LOGGING=0 \
  python "$PROJECT_DIR/vllm_infer.py" \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER" \
    --dataset "$DATASET_KEY" \
    --template ${TEMPLATE:-"llama3"} \
    --dataset_dir "$DATASET_DIR" \
    --save_name "$OUTPUT_FILE"
}

run_llm_sft() {
  local DATASET_KEY="$1"
  local OUTDIR="$2"
  local LOGDIR="$3"
  local LR="$4"
  local EPOCHS="$5"
  local ADAPTER="$6"  # optional: continue from adapter
  
  mkdir -p "$OUTDIR" "$LOGDIR"
  
  local ADAPTER_ARGS=""
  if [ -n "$ADAPTER" ]; then
    ADAPTER_ARGS="--adapter_name_or_path $ADAPTER --create_new_adapter"
  fi
  
  llamafactory-cli train \
    --stage sft --do_train \
    --model_name_or_path "$BASE_MODEL_PATH" \
    $ADAPTER_ARGS \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$DATASET_KEY" \
    --template ${TEMPLATE:-"llama3"} \
    --finetuning_type lora \
    --lora_rank ${LORA_RANK:-8} --lora_alpha ${LORA_ALPHA:-16} --lora_target all \
    --output_dir "$OUTDIR" --overwrite_cache --overwrite_output_dir \
    --cutoff_len 2048 --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${BATCH_SIZE_SFT:-4} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-2} \
    --lr_scheduler_type cosine --logging_steps 20 --save_steps 500 \
    --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --plot_loss --bf16 --save_total_limit 3 \
    --logging_dir "$LOGDIR" 2>&1 | tee "$LOGDIR/train.log"
}


# ===========================================================================
# STAGE 1-3: Initialize GNN and LLM (skip if SKIP_INIT=1)
# ===========================================================================
if [ "${SKIP_INIT:-0}" = "1" ]; then
  echo ""
  echo "--- SKIP_INIT=1: Reusing existing round 0 checkpoints ---"

  # Verify round 0 exists
  ROUND0_DIR="$RUN_DIR/round0"
  SFT_OUTDIR="$ROUND0_DIR/sft/model"

  if [ ! -f "$GNN_MODEL_PATH" ]; then
    echo "ERROR: GNN model not found at $GNN_MODEL_PATH"
    exit 1
  fi
  echo "  GNN: $GNN_MODEL_PATH"

  CURRENT_LLM_ADAPTER="$SFT_OUTDIR"
  if [ -d "$SFT_OUTDIR" ]; then
    LATEST=$(ls -dt "$SFT_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
    [ -n "$LATEST" ] && CURRENT_LLM_ADAPTER="$LATEST"
  fi
  if [ ! -d "$CURRENT_LLM_ADAPTER" ]; then
    echo "ERROR: LLM adapter not found at $SFT_OUTDIR"
    exit 1
  fi
  echo "  LLM: $CURRENT_LLM_ADAPTER"
  INIT_LLM_ADAPTER="$CURRENT_LLM_ADAPTER"

  # Make sure dataset_info.json has the SFT entries
  _DSINFO="$DATASET_INFO_FILE" _PREFIX="$SFT_DATASET_PREFIX" python - <<'SKIP_EOF'
import json, os
p = os.environ["_DSINFO"]
prefix = os.environ["_PREFIX"]
info = json.load(open(p)) if os.path.exists(p) else {}
for split in ["train", "val", "test", "unlabeled"]:
    key = f"{prefix}_{split}"
    if key not in info:
        info[key] = {
            "file_name": f"{key}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"}
        }
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
SKIP_EOF

else
# --- Normal initialization: Stage 1, 2, 3 ---

# ===========================================================================
# STAGE 1: Create SFT Dataset
# ===========================================================================
echo ""
echo "--- Stage 1: Create SFT Dataset ---"
cd "$PROJECT_DIR"
python create_sft.py \
  --dataset "$DATASET" \
  --output "$DATASET_DIR/${DATASET}_sft.json" \
  --shots "$SHOT_COUNT" \
  --seed "$SEED" \
  --path_prefix "." \
  --use_neighbor_info "${USE_NEIGHBOR_INFO:-1}"

python - <<EOF
import json, os
p = "$DATASET_INFO_FILE"
prefix = "$SFT_DATASET_PREFIX"
os.makedirs(os.path.dirname(p), exist_ok=True)
info = json.load(open(p)) if os.path.exists(p) else {}
for split in ["train", "val", "test", "unlabeled"]:
    key = f"{prefix}_{split}"
    info[key] = {
        "file_name": f"{key}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"}
    }
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
print("Dataset info updated for SFT")
EOF


# ===========================================================================
# STAGE 2: Train Initial GNN
# ===========================================================================
echo ""
echo "--- Stage 2: Train Initial GNN ---"
cd "$PROJECT_DIR/GNN"
python main.py \
  --dataset "$DATASET" \
  --shots "$SHOT_COUNT" \
  --gnn_type "${GNN_TYPE:-GCN}" \
  --hidden_dim ${GNN_HIDDEN_DIM:-64} \
  --n_layers ${GNN_LAYERS:-2} \
  --dropout ${GNN_DROPOUT:-0.5} \
  --learning_rate ${GNN_LR:-1e-2} \
  --epochs ${GNN_EPOCHS:-500} \
  --patience ${GNN_PATIENCE:-100} \
  --seed "$SEED" \
  --run_times 1 \
  --device "${DEVICE:-cuda:0}"
cd "$PROJECT_DIR"

SRC_GNN="$PROJECT_DIR/results/GNN/${DATASET}_${SHOT_COUNT}_shot_best_model_run0.pt"
cp "$SRC_GNN" "$GNN_MODEL_PATH"
echo "Initial GNN model: $GNN_MODEL_PATH"


# ===========================================================================
# STAGE 3: Train Initial LLM (SFT) — using llamafactory-cli
# ===========================================================================
echo ""
echo "--- Stage 3: Train Initial LLM (SFT) ---"

ROUND0_DIR="$RUN_DIR/round0"
SFT_LOG_DIR="$ROUND0_DIR/sft/logs"
SFT_OUTDIR="$ROUND0_DIR/sft/model"
mkdir -p "$SFT_LOG_DIR" "$SFT_OUTDIR"

llamafactory-cli train \
  --stage sft --do_train \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --dataset "${SFT_DATASET_PREFIX}_train" \
  --template ${TEMPLATE:-"llama3"} \
  --finetuning_type lora \
  --lora_rank ${LORA_RANK:-8} --lora_alpha ${LORA_ALPHA:-16} --lora_target all \
  --output_dir "$SFT_OUTDIR" --overwrite_cache --overwrite_output_dir \
  --cutoff_len 2048 --preprocessing_num_workers 16 \
  --per_device_train_batch_size ${BATCH_SIZE_SFT:-4} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-2} \
  --lr_scheduler_type cosine --logging_steps 20 --save_steps 500 \
  --learning_rate ${LEARNING_RATE_SFT:-5e-5} --num_train_epochs ${EPOCHS_SFT:-3} \
  --plot_loss --bf16 --save_total_limit 3 \
  --logging_dir "$SFT_LOG_DIR" 2>&1 | tee "$SFT_LOG_DIR/train.log"

CURRENT_LLM_ADAPTER="$SFT_OUTDIR"
if [ -d "$SFT_OUTDIR" ]; then
  LATEST=$(ls -dt "$SFT_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
  [ -n "$LATEST" ] && CURRENT_LLM_ADAPTER="$LATEST"
fi
echo "Initial LLM adapter: $CURRENT_LLM_ADAPTER"
INIT_LLM_ADAPTER="$CURRENT_LLM_ADAPTER"  # Save for Co-Teaching: each round starts from here

# Quick eval: test initial SFT on a small sample
echo ""
echo "--- Stage 3.5: Quick Eval of Initial SFT ---"
INIT_PRED_FILE="$ROUND0_DIR/sft_test_predictions.jsonl"
run_llm_inference "$CURRENT_LLM_ADAPTER" "${SFT_DATASET_PREFIX}_test" "$INIT_PRED_FILE"

cd "$PROJECT_DIR"
python evaluate_predictions.py \
  --dataset "$DATASET" \
  --pred_file "$INIT_PRED_FILE" \
  --output_dir "$ROUND0_DIR/eval" \
  --model_name "initial_sft" \
  --path_prefix "."


fi  # end SKIP_INIT


# ===========================================================================
# STAGE 4: Prepare unlabeled pool (all unlabeled nodes, sample each round)
# ===========================================================================
echo ""
echo "--- Stage 4: Prepare Unlabeled Node Pool ---"
cd "$PROJECT_DIR"

FULL_NODE_IDS_FILE="$DATASET_DIR/${DATASET}_${SHOT_COUNT}_shot_unlabeled_node_ids.json"
UNLABELED_FILE="$DATASET_DIR/${SFT_DATASET_PREFIX}_unlabeled.json"
BATCH_SIZE=${BATCH_SIZE_POOL:-1500}
POOL_MODE=${POOL_MODE:-random}

# If POOL_MODE=influential, compute GAJ-style top-K influential node pool
# and use it as the unlabeled pool. The per-round random sampler below will
# draw BATCH_SIZE from this pool (if BATCH_SIZE >= K, all K are used every round,
# which matches GAJ's fixed-pool behavior).
if [ "$POOL_MODE" = "influential" ]; then
  INFLUENTIAL_TOPK=${INFLUENTIAL_TOPK:-1500}
  INFLUENTIAL_MAX_SUBGRAPH_NODES=${INFLUENTIAL_MAX_SUBGRAPH_NODES:-3000}
  INFLUENTIAL_FILE="$RUN_DIR/influential_nodes_top${INFLUENTIAL_TOPK}.json"

  if [ ! -f "$INFLUENTIAL_FILE" ]; then
    echo "--- Stage 4a: Computing top-${INFLUENTIAL_TOPK} influential nodes (GAJ-style) ---"
    cd "$PROJECT_DIR"
    python "$PROJECT_DIR/select_influential_nodes.py" \
      --dataset "$DATASET" \
      --shots "$SHOT_COUNT" \
      --seed "$SEED" \
      --k "$INFLUENTIAL_TOPK" \
      --method auto \
      --max_subgraph_nodes "$INFLUENTIAL_MAX_SUBGRAPH_NODES" \
      --path_prefix "." \
      --output_file "$INFLUENTIAL_FILE"
  else
    echo "--- Stage 4a: Reusing cached influential nodes at $INFLUENTIAL_FILE ---"
  fi

  # Build a filtered UNLABELED_FILE aligned with the influential pool, so that
  # sample_and_build_dataset can keep its index-based join logic unchanged.
  NODE_IDS_FILE="$RUN_DIR/pool_node_ids.json"
  POOL_UNLABELED_FILE="$RUN_DIR/pool_unlabeled.json"
  python - <<PYEOF
import json
full_ids = json.load(open("$FULL_NODE_IDS_FILE"))["selected_node_ids"]
unlabeled = json.load(open("$UNLABELED_FILE"))
influential = json.load(open("$INFLUENTIAL_FILE"))["selected_node_ids"]
infl_set = set(int(x) for x in influential)
id2conv = {int(nid): unlabeled[i] for i, nid in enumerate(full_ids)}
kept_ids = [int(nid) for nid in influential if int(nid) in id2conv]
kept_convs = [id2conv[nid] for nid in kept_ids]
json.dump({"selected_node_ids": kept_ids}, open("$NODE_IDS_FILE", "w"), indent=2)
json.dump(kept_convs, open("$POOL_UNLABELED_FILE", "w"), ensure_ascii=False, indent=2)
print(f"Influential pool: {len(kept_ids)} nodes (from top-{len(influential)})")
PYEOF
  UNLABELED_FILE="$POOL_UNLABELED_FILE"
else
  NODE_IDS_FILE="$FULL_NODE_IDS_FILE"
fi

echo "Total unlabeled pool: $(python -c "import json; print(len(json.load(open('$NODE_IDS_FILE'))['selected_node_ids']))")"
echo "Mini-batch size per round: $BATCH_SIZE (POOL_MODE=$POOL_MODE)"

CURRENT_GNN_MODEL="$GNN_MODEL_PATH"


# ===========================================================================
# Helper: Random sample a mini-batch and build inference dataset
# ===========================================================================
sample_and_build_dataset() {
  local ROUND_TAG="$1"
  local BATCH_SZ="$2"
  local ROUND_DIR_LOCAL="$3"

  local SAMPLE_FILE="$ROUND_DIR_LOCAL/sampled_nodes.json"
  local DATASET_NAME="${SFT_DATASET_PREFIX}_batch_r${ROUND_TAG}"
  local DATASET_FILE="$DATASET_DIR/${DATASET_NAME}.json"
  local ORDERED_FILE="$ROUND_DIR_LOCAL/sampled_nodes_ordered.json"

  cd "$PROJECT_DIR"
  python - <<PYEOF
import json, os, sys, random

random.seed(${SEED} + (${ROUND_TAG} - 1) // 2)  # consecutive round pairs share same batch

with open("$NODE_IDS_FILE") as f:
    all_ids = json.load(f)['selected_node_ids']
with open("$UNLABELED_FILE") as f:
    unlabeled = json.load(f)

# Random sample mini-batch
sampled_ids = random.sample(all_ids, min($BATCH_SZ, len(all_ids)))
sampled_set = set(sampled_ids)

# Build dataset
filtered, ordered_ids = [], []
for i, nid in enumerate(all_ids):
    if nid in sampled_set:
        filtered.append(unlabeled[i])
        ordered_ids.append(nid)

with open("$DATASET_FILE", 'w') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
with open("$SAMPLE_FILE", 'w') as f:
    json.dump({"selected_node_ids": sampled_ids}, f, indent=2)
with open("$ORDERED_FILE", 'w') as f:
    json.dump({"selected_node_ids": ordered_ids}, f, indent=2)

info_path = "$DATASET_INFO_FILE"
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
info["$DATASET_NAME"] = {
    "file_name": os.path.basename("$DATASET_FILE"),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(info_path, 'w'), indent=2, ensure_ascii=False)
print(f"Round {$ROUND_TAG}: sampled {len(filtered)} nodes from {len(all_ids)} unlabeled", file=sys.stderr)
PYEOF

  echo "$DATASET_NAME"
}


# ===========================================================================
# STAGE 5: Co-Teaching Rounds (random batch + cross-select + continue train)
# ===========================================================================
for ROUND in $(seq 1 $NUM_ROUNDS); do

echo ""
echo "================================================================"
echo "=== Co-Teaching Round $ROUND / $NUM_ROUNDS ==="
echo "================================================================"

ROUND_DIR="$RUN_DIR/round${ROUND}"
mkdir -p "$ROUND_DIR"

# ----- Step 0: Random sample a mini-batch from all unlabeled nodes -----
echo "--- Round $ROUND Step 0: Random Sample $BATCH_SIZE Nodes ---"
SELECTED_DATASET_NAME=$(sample_and_build_dataset "$ROUND" "$BATCH_SIZE" "$ROUND_DIR")
ORDERED_NODES_FILE="$ROUND_DIR/sampled_nodes_ordered.json"

ALPHA=$(python -c "
base = ${GNN_TEACH_ALPHA:-0.3}
increment = 0.05
print(min(base + ($ROUND - 1) * increment, 0.7))
")
echo "GNN teaching alpha: $ALPHA"


# ----- 5a: LLM predicts on selected nodes -----
echo ""
echo "--- Round $ROUND Step 1: LLM Predicts on Selected Nodes ---"
LLM_PRED_FILE="$ROUND_DIR/${RUN_ID}_round${ROUND}_llm_preds.jsonl"
run_llm_inference "$CURRENT_LLM_ADAPTER" "$SELECTED_DATASET_NAME" "$LLM_PRED_FILE"


# ----- 5b: Cross-Filter selection (bidirectional) -----
echo ""
echo "--- Round $ROUND Step 2: Cross-Filter Selection ---"
cd "$PROJECT_DIR"

SFT_COTEACH_FILE="$DATASET_DIR/${RUN_ID}_round${ROUND}_gnn_selects_for_llm.json"
GNN_PSEUDO_FILE="$ROUND_DIR/${RUN_ID}_round${ROUND}_llm_selects_for_gnn.json"

python create_co_teaching_data.py \
  --dataset "$DATASET" \
  --selected_nodes_path "$ORDERED_NODES_FILE" \
  --pretrained_model "$CURRENT_GNN_MODEL" \
  --llm_predictions "$LLM_PRED_FILE" \
  --sft_output_path "$SFT_COTEACH_FILE" \
  --gnn_pseudo_label_path "$GNN_PSEUDO_FILE" \
  --round "$ROUND" \
  --shots "$SHOT_COUNT" \
  --gnn_type "${GNN_TYPE:-GCN}" \
  --hidden_dim ${GNN_HIDDEN_DIM:-64} \
  --n_layers ${GNN_LAYERS:-2} \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:0}" \
  --path_prefix "." \
  --retrain_on_agreed 1 \
  --anchor_repeat "${ANCHOR_REPEAT:-3}" \
  --use_cumulative "${USE_CUMULATIVE:-0}" \
  --num_rounds "$NUM_ROUNDS" \
  --rt_min "${RT_MIN:-0.2}" \
  --rt_max "${RT_MAX:-0.6}" \
  --use_neighbor_info "${USE_NEIGHBOR_INFO:-1}"


# ----- 5c: GNN continues training with LLM pseudo-labels -----
echo ""
echo "--- Round $ROUND Step 3: Continue GNN with Pseudo-Labels ---"
UPDATED_GNN_MODEL="$ROUND_DIR/gnn_round${ROUND}.pt"

if [ "${USE_CUMULATIVE:-1}" = "1" ] && [ "$ROUND" -gt 1 ]; then
  # Cumulative: merge historical agreed + current round full
  # Historical agreed nodes get updated with latest label if available in current round
  MERGED_PSEUDO_FILE="$ROUND_DIR/${RUN_ID}_round${ROUND}_merged_pseudo.json"
  python - <<EOF
import json, os, glob

run_dir = "$RUN_DIR"
current_round = $ROUND

# First: collect all historical agreed labels
historical_agreed = {}
for r in range(1, current_round):
    pattern = os.path.join(run_dir, "round" + str(r), "*_llm_selects_for_gnn_agreed.json")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        with open(candidates[0]) as f:
            pl = json.load(f).get("pseudo_labels", {})
        historical_agreed.update(pl)
        print("  Round {} (agreed): {} labels".format(r, len(pl)))

# Then: load current round full labels
current_full = {}
pattern = os.path.join(run_dir, "round" + str(current_round), "*_llm_selects_for_gnn.json")
candidates = [c for c in sorted(glob.glob(pattern)) if '_agreed' not in c]
if candidates:
    with open(candidates[0]) as f:
        current_full = json.load(f).get("pseudo_labels", {})
    print("  Round {} (current full): {} labels".format(current_round, len(current_full)))

# Merge: start with historical, then update with current (latest wins)
merged = {}
merged.update(historical_agreed)
# Update historical agreed nodes with current round label if available
n_updated = 0
for nid in list(merged.keys()):
    if nid in current_full:
        merged[nid] = current_full[nid]
        n_updated += 1
# Add remaining current round nodes not in historical
for nid, lid in current_full.items():
    if nid not in merged:
        merged[nid] = lid

print("  Historical agreed: {}, current full: {}, updated: {}".format(
    len(historical_agreed), len(current_full), n_updated))
print("  Total merged: {} unique pseudo-labeled nodes".format(len(merged)))
with open("$MERGED_PSEUDO_FILE", 'w') as f:
    json.dump({"pseudo_labels": merged}, f)
EOF
  GNN_PSEUDO_INPUT="$MERGED_PSEUDO_FILE"
else
  GNN_PSEUDO_INPUT="$GNN_PSEUDO_FILE"
fi

python train_gnn_with_pseudo_labels.py \
  --dataset "$DATASET" \
  --pseudo_label_path "$GNN_PSEUDO_INPUT" \
  --pretrained_model "$CURRENT_GNN_MODEL" \
  --save_path "$UPDATED_GNN_MODEL" \
  --shots "$SHOT_COUNT" \
  --gnn_type "${GNN_TYPE:-GCN}" \
  --hidden_dim ${GNN_HIDDEN_DIM:-64} \
  --n_layers ${GNN_LAYERS:-2} \
  --dropout ${GNN_DROPOUT:-0.5} \
  --learning_rate ${GNN_TEACH_LR:-1e-3} \
  --epochs ${GNN_TEACH_EPOCHS:-50} \
  --patience ${GNN_PATIENCE:-100} \
  --alpha "$ALPHA" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:0}" \
  --path_prefix "."

CURRENT_GNN_MODEL="$UPDATED_GNN_MODEL"


# ----- 5d: LLM continues training with GNN labels -----
echo ""
echo "--- Round $ROUND Step 4: Continue LLM with GNN Pseudo-Labels ---"
SFT_COTEACH_KEY="${RUN_ID}_round${ROUND}_gnn_selects_for_llm"
SFT_COTEACH_KEY_AGREED="${RUN_ID}_round${ROUND}_gnn_selects_for_llm_agreed"

# Register datasets
python - <<EOF
import json, os
p = "$DATASET_INFO_FILE"
info = json.load(open(p)) if os.path.exists(p) else {}
info["$SFT_COTEACH_KEY"] = {
    "file_name": os.path.basename("$SFT_COTEACH_FILE"),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
agreed_file = os.path.basename("$SFT_COTEACH_FILE").replace('.json', '_agreed.json')
if os.path.exists(os.path.join("$DATASET_DIR", agreed_file)):
    info["$SFT_COTEACH_KEY_AGREED"] = {
        "file_name": agreed_file,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"}
    }
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
EOF

# Build SFT dataset key
if [ "${USE_CUMULATIVE:-1}" = "1" ] && [ "$ROUND" -gt 1 ]; then
  # On-the-fly merge: historical agreed + current full, all with latest labels
  MERGED_SFT_FILE="$DATASET_DIR/${RUN_ID}_round${ROUND}_cumulative_sft.json"
  MERGED_SFT_KEY="${RUN_ID}_round${ROUND}_cumulative_sft"

  python - <<MERGE_EOF
import json, os, glob

run_dir = "$RUN_DIR"
dataset_dir = "$DATASET_DIR"
current_round = $ROUND

# Step 1: Collect all pseudo-label entries from historical agreed files
# key = node_id, value = full SFT entry (latest round wins)
anchors = {}    # node_id -> entry (ground-truth, never changes)
pseudo = {}     # node_id -> entry (latest label wins)

for r in range(1, current_round):
    pattern = os.path.join(dataset_dir, "*_round" + str(r) + "_gnn_selects_for_llm_agreed.json")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        with open(candidates[0]) as f:
            data = json.load(f)
        n_a, n_p = 0, 0
        for item in data:
            nid = item.get("node_id")
            if nid is None:
                continue
            if item.get("is_anchor", False):
                anchors[nid] = item
                n_a += 1
            else:
                pseudo[nid] = item  # later round overwrites
                n_p += 1
        print("  Round {} (agreed): {} anchors, {} pseudo".format(r, n_a, n_p))

# Step 2: Current round full — overwrites historical for same node_id
current_file = os.path.join(dataset_dir, "${RUN_ID}_round${ROUND}_gnn_selects_for_llm.json")
if os.path.exists(current_file):
    with open(current_file) as f:
        data = json.load(f)
    n_a, n_p = 0, 0
    for item in data:
        nid = item.get("node_id")
        if nid is None:
            continue
        if item.get("is_anchor", False):
            anchors[nid] = item
            n_a += 1
        else:
            pseudo[nid] = item  # current round overwrites all
            n_p += 1
    print("  Round {} (current full): {} anchors, {} pseudo".format(current_round, n_a, n_p))

# Step 3: Assemble final dataset
# Anchors: keep all (including repeats from anchor_repeat)
# Pseudo: deduplicated by node_id, latest label
anchor_list = list(anchors.values())
pseudo_list = list(pseudo.values())

# Re-apply anchor_repeat: anchors dict has unique entries, repeat them
anchor_repeat = ${ANCHOR_REPEAT:-3}
final_anchors = anchor_list * anchor_repeat

merged = final_anchors + pseudo_list
print("  Merged: {} anchors ({}x{}) + {} pseudo = {} total".format(
    len(final_anchors), len(anchor_list), anchor_repeat, len(pseudo_list), len(merged)))

with open("$MERGED_SFT_FILE", 'w') as f:
    json.dump(merged, f, indent=2)
MERGE_EOF

  # Register merged dataset
  python - <<REG_EOF
import json, os
p = "$DATASET_INFO_FILE"
info = json.load(open(p)) if os.path.exists(p) else {}
info["$MERGED_SFT_KEY"] = {
    "file_name": os.path.basename("$MERGED_SFT_FILE"),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
REG_EOF

  SFT_TRAIN_KEY="$MERGED_SFT_KEY"
  echo "  Cumulative SFT (merged): $SFT_TRAIN_KEY"
else
  SFT_TRAIN_KEY="$SFT_COTEACH_KEY"
  echo "  Single-round SFT: $SFT_TRAIN_KEY"
fi

SFT_ROUND_OUTDIR="$ROUND_DIR/sft/model"
SFT_ROUND_LOG="$ROUND_DIR/sft/logs"

run_llm_sft "$SFT_TRAIN_KEY" "$SFT_ROUND_OUTDIR" "$SFT_ROUND_LOG" \
  "${LLM_TEACH_LR:-2e-5}" "${LLM_TEACH_EPOCHS:-8}" "$CURRENT_LLM_ADAPTER"

CURRENT_LLM_ADAPTER="$SFT_ROUND_OUTDIR"
if [ -d "$SFT_ROUND_OUTDIR" ]; then
  LATEST=$(ls -dt "$SFT_ROUND_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
  [ -n "$LATEST" ] && CURRENT_LLM_ADAPTER="$LATEST"
fi
echo "Round $ROUND LLM adapter: $CURRENT_LLM_ADAPTER"


# ----- 5e: Evaluate & Log Progress -----
echo ""
echo "--- Round $ROUND Step 5: Evaluate ---"
ROUND_PRED_FILE="$ROUND_DIR/test_predictions.jsonl"
run_llm_inference "$CURRENT_LLM_ADAPTER" "${SFT_DATASET_PREFIX}_test" "$ROUND_PRED_FILE"

cd "$PROJECT_DIR"
python evaluate_predictions.py \
  --dataset "$DATASET" \
  --pred_file "$ROUND_PRED_FILE" \
  --output_dir "$ROUND_DIR/eval" \
  --model_name "round${ROUND}" \
  --path_prefix "."

# Append to progress.csv (one file to watch)
PROGRESS_FILE="$RUN_DIR/progress.csv"
python - <<PYEOF
import json, os, glob

progress = "$PROGRESS_FILE"
round_num = $ROUND

# Write header if first round
if not os.path.exists(progress) or round_num == 1:
    with open(progress, 'w') as f:
        f.write("round,gnn_test_acc,gnn_test_f1,llm_test_acc,llm_test_f1,R_t,n_agreed,batch_size,gnn_selected,llm_selected,gnn_pseudo_acc,llm_pseudo_acc,ema_gnn_acc,ema_gnn_f1\n")

# GNN metrics
gnn_acc, gnn_f1 = "", ""
ema_gnn_acc, ema_gnn_f1 = "", ""
gnn_metrics_file = "$UPDATED_GNN_MODEL".replace('.pt', '_metrics.json')
if os.path.exists(gnn_metrics_file):
    with open(gnn_metrics_file) as f:
        gm = json.load(f)
    gnn_acc = gm.get("test_acc", "")
    gnn_f1 = gm.get("test_macro_f1", "")
    ema_gnn_acc = gm.get("ema_test_acc", "")
    ema_gnn_f1 = gm.get("ema_test_f1", "")

# Cross-selection stats
stats_files = glob.glob(os.path.join("$ROUND_DIR", "*_stats.json"))
rt, n_agreed, batch_sz = "", "", ""
gnn_sel, llm_sel, gnn_pa, llm_pa = "", "", "", ""
for sf in stats_files:
    with open(sf) as f:
        s = json.load(f)
    rt = s.get("select_ratio_Rt", s.get("agreed_rate", ""))
    n_agreed = s.get("n_agreed", "")
    batch_sz = s.get("batch_size", "")
    if "gnn_selects_for_llm" in s:
        gnn_sel = s["gnn_selects_for_llm"].get("n_selected", "")
        gnn_pa = s["gnn_selects_for_llm"].get("pseudo_label_accuracy", "")
    if "llm_selects_for_gnn" in s:
        llm_sel = s["llm_selects_for_gnn"].get("n_selected", "")
        llm_pa = s["llm_selects_for_gnn"].get("pseudo_label_accuracy", "")

# LLM metrics
llm_acc, llm_f1 = "", ""
llm_metrics = os.path.join("$ROUND_DIR", f"eval/round{round_num}/metrics.json")
if os.path.exists(llm_metrics):
    with open(llm_metrics) as f:
        m = json.load(f)
    llm_acc = m.get("accuracy", "")
    llm_f1 = m.get("macro_f1", "")

with open(progress, 'a') as f:
    f.write(f"{round_num},{gnn_acc},{gnn_f1},{llm_acc},{llm_f1},{rt},{n_agreed},{batch_sz},{gnn_sel},{llm_sel},{gnn_pa},{llm_pa},{ema_gnn_acc},{ema_gnn_f1}\n")

print(f"Round {round_num}: GNN Acc={gnn_acc} (EMA={ema_gnn_acc}) | LLM Acc={llm_acc} F1={llm_f1} | R(t)={rt} agreed={n_agreed}")
PYEOF

echo "Progress saved to: $PROGRESS_FILE"


# ----- 5f: DPO on preference data (after even rounds) -----
if [ "${USE_DPO:-1}" = "1" ] && [ $((ROUND % 2)) -eq 0 ] && [ "$ROUND" -ge 2 ]; then
  ODD_ROUND=$((ROUND - 1))
  echo ""
  echo "--- Round $ROUND Step 6: Construct Preference Data (Round $ODD_ROUND → $ROUND) ---"

  ODD_ROUND_DIR="$RUN_DIR/round${ODD_ROUND}"
  EVEN_ROUND_DIR="$ROUND_DIR"
  PREF_FILE="$DATASET_DIR/${RUN_ID}_round${ROUND}_preference.json"
  PREF_KEY="${RUN_ID}_round${ROUND}_preference"

  # Find LLM prediction files (batch predictions, not test)
  ODD_PRED="$ODD_ROUND_DIR/${RUN_ID}_round${ODD_ROUND}_llm_preds.jsonl"
  EVEN_PRED="$EVEN_ROUND_DIR/${RUN_ID}_round${ROUND}_llm_preds.jsonl"

  # Find GNN models (the one BEFORE training = pretrained input to that round)
  ODD_GNN="$ODD_ROUND_DIR/gnn_round${ODD_ROUND}.pt"
  EVEN_GNN="$EVEN_ROUND_DIR/gnn_round${ROUND}.pt"

  if [ -n "$ODD_PRED" ] && [ -n "$EVEN_PRED" ] && [ -f "$ODD_GNN" ] && [ -f "$EVEN_GNN" ]; then
    cd "$PROJECT_DIR"
    python create_preference_data.py \
      --dataset "$DATASET" \
      --odd_round_dir "$ODD_ROUND_DIR" \
      --even_round_dir "$EVEN_ROUND_DIR" \
      --odd_round_predictions "$ODD_PRED" \
      --even_round_predictions "$EVEN_PRED" \
      --odd_round_gnn_model "$ODD_GNN" \
      --even_round_gnn_model "$EVEN_GNN" \
      --output_path "$PREF_FILE" \
      --odd_round "$ODD_ROUND" \
      --even_round "$ROUND" \
      --shots "$SHOT_COUNT" \
      --gnn_type "${GNN_TYPE:-GCN}" \
      --hidden_dim ${GNN_HIDDEN_DIM:-64} \
      --n_layers ${GNN_LAYERS:-2} \
      --seed "$SEED" \
      --device "${DEVICE:-cuda:0}" \
      --path_prefix "."

    # Check if we got enough preference pairs
    N_PAIRS=$(python -c "import json; print(len(json.load(open('$PREF_FILE'))))" 2>/dev/null || echo "0")
    echo "  Preference pairs: $N_PAIRS"

    if [ "$N_PAIRS" -gt 10 ]; then
      # Register preference dataset
      python - <<DPO_REG_EOF
import json, os
p = "$DATASET_INFO_FILE"
info = json.load(open(p)) if os.path.exists(p) else {}
info["$PREF_KEY"] = {
    "file_name": os.path.basename("$PREF_FILE"),
    "ranking": True,
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"}
}
json.dump(info, open(p, 'w'), indent=2, ensure_ascii=False)
DPO_REG_EOF

      echo "--- Round $ROUND Step 6: DPO Training on Preference Data ---"
      DPO_OUTDIR="$ROUND_DIR/dpo/model"
      DPO_LOG="$ROUND_DIR/dpo/logs"
      mkdir -p "$DPO_OUTDIR" "$DPO_LOG"

      DPO_ADAPTER_ARGS=""
      if [ -n "$CURRENT_LLM_ADAPTER" ]; then
        DPO_ADAPTER_ARGS="--adapter_name_or_path $CURRENT_LLM_ADAPTER --create_new_adapter"
      fi

      llamafactory-cli train \
        --stage dpo --do_train \
        --model_name_or_path "$BASE_MODEL_PATH" \
        $DPO_ADAPTER_ARGS \
        --dataset_dir "$DATASET_DIR" \
        --dataset "$PREF_KEY" \
        --template ${TEMPLATE:-"llama3"} \
        --finetuning_type lora \
        --lora_rank ${LORA_RANK:-8} --lora_alpha ${LORA_ALPHA:-16} --lora_target all \
        --output_dir "$DPO_OUTDIR" --overwrite_cache --overwrite_output_dir \
        --cutoff_len 2048 --preprocessing_num_workers 16 \
        --per_device_train_batch_size ${BATCH_SIZE_DPO:-2} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS:-2} \
        --lr_scheduler_type cosine --logging_steps 10 --save_steps 500 \
        --learning_rate "${DPO_LR:-5e-6}" --num_train_epochs "${DPO_EPOCHS:-1}" \
        --pref_loss ${PREF_LOSS:-sigmoid} --pref_beta ${DPO_BETA:-0.1} \
        --plot_loss --bf16 --save_total_limit 1 \
        --logging_dir "$DPO_LOG" 2>&1 | tee "$DPO_LOG/train.log"

      # Update LLM adapter to DPO output
      if [ -d "$DPO_OUTDIR" ]; then
        DPO_LATEST=$(ls -dt "$DPO_OUTDIR"/checkpoint-* 2>/dev/null | head -n 1)
        if [ -n "$DPO_LATEST" ]; then
          CURRENT_LLM_ADAPTER="$DPO_LATEST"
          echo "  DPO adapter updated: $CURRENT_LLM_ADAPTER"
        fi
      fi
    else
      echo "  Too few preference pairs ($N_PAIRS), skipping DPO"
    fi
  else
    echo "  Missing prediction files or GNN models, skipping DPO"
    echo "  ODD_PRED=$ODD_PRED EVEN_PRED=$EVEN_PRED"
    echo "  ODD_GNN=$ODD_GNN EVEN_GNN=$EVEN_GNN"
  fi
fi

done


# ===========================================================================
# STAGE 6: Final Summary — save all paths and metrics
# ===========================================================================
echo ""
echo "================================================================"
echo "=== Cross-Filter Pipeline Completed ==="
echo "================================================================"
echo "Run ID: $RUN_ID"
echo "Results: $RUN_DIR"
echo ""

# Print all saved model paths
echo "--- Saved Models ---"
echo "  GNN initial:  $GNN_MODEL_PATH"
echo "  LLM initial:  $INIT_LLM_ADAPTER"
for ROUND in $(seq 1 $NUM_ROUNDS); do
  echo "  GNN round $ROUND:  $RUN_DIR/round${ROUND}/gnn_round${ROUND}.pt"
  R_ADAPTER="$RUN_DIR/round${ROUND}/sft/model"
  LATEST_R=$(ls -dt "$R_ADAPTER"/checkpoint-* 2>/dev/null | head -n 1)
  [ -n "$LATEST_R" ] && R_ADAPTER="$LATEST_R"
  echo "  LLM round $ROUND:  $R_ADAPTER"
done

echo ""
echo "--- Round-by-round Results ---"

# Collect and print all metrics, also save summary JSON
python - <<PYEOF
import json, os, glob

run_dir = "$RUN_DIR"
num_rounds = $NUM_ROUNDS
summary = {"run_id": "$RUN_ID", "dataset": "$DATASET", "shots": $SHOT_COUNT, "seed": $SEED, "rounds": []}

# Initial SFT eval
init_metrics_file = os.path.join(run_dir, "round0/eval/initial_sft/metrics.json")
if os.path.exists(init_metrics_file):
    with open(init_metrics_file) as f:
        m = json.load(f)
    print(f"  Round 0 (initial SFT): Acc={m.get('accuracy', 0):.4f}  Macro-F1={m.get('macro_f1', 0):.4f}  Samples={m.get('total_samples', 0)}")
    summary["initial_sft"] = m

# Co-teaching rounds
for r in range(1, num_rounds + 1):
    entry = {"round": r}
    
    # LLM metrics
    llm_metrics = os.path.join(run_dir, f"round{r}/eval/round{r}/metrics.json")
    if os.path.exists(llm_metrics):
        with open(llm_metrics) as f:
            m = json.load(f)
        entry["llm_accuracy"] = m.get("accuracy", 0)
        entry["llm_macro_f1"] = m.get("macro_f1", 0)
        entry["llm_samples"] = m.get("total_samples", 0)
    
    # Cross-selection stats
    stats_files = glob.glob(os.path.join(run_dir, f"round{r}/*_stats.json"))
    for sf in stats_files:
        with open(sf) as f:
            s = json.load(f)
        entry["select_ratio"] = s.get("select_ratio", 0)
        if "gnn_selects_for_llm" in s:
            entry["gnn_selected"] = s["gnn_selects_for_llm"].get("n_selected", 0)
            entry["gnn_pseudo_acc"] = s["gnn_selects_for_llm"].get("pseudo_label_accuracy", 0)
        if "llm_selects_for_gnn" in s:
            entry["llm_selected"] = s["llm_selects_for_gnn"].get("n_selected", 0)
            entry["llm_pseudo_acc"] = s["llm_selects_for_gnn"].get("pseudo_label_accuracy", 0)
    
    # GNN model path
    entry["gnn_model"] = os.path.join(run_dir, f"round{r}/gnn_round{r}.pt")
    adapter_dir = os.path.join(run_dir, f"round{r}/sft/model")
    ckpts = sorted(glob.glob(os.path.join(adapter_dir, "checkpoint-*")))
    entry["llm_adapter"] = ckpts[-1] if ckpts else adapter_dir
    
    summary["rounds"].append(entry)
    
    llm_acc = entry.get("llm_accuracy", "N/A")
    llm_f1 = entry.get("llm_macro_f1", "N/A")
    gnn_sel = entry.get("gnn_selected", "?")
    llm_sel = entry.get("llm_selected", "?")
    gnn_pa = entry.get("gnn_pseudo_acc", "?")
    llm_pa = entry.get("llm_pseudo_acc", "?")
    sr = entry.get("select_ratio", "?")
    
    acc_str = f"{llm_acc:.4f}" if isinstance(llm_acc, float) else str(llm_acc)
    f1_str = f"{llm_f1:.4f}" if isinstance(llm_f1, float) else str(llm_f1)
    gnn_pa_str = f"{gnn_pa:.4f}" if isinstance(gnn_pa, float) else str(gnn_pa)
    llm_pa_str = f"{llm_pa:.4f}" if isinstance(llm_pa, float) else str(llm_pa)
    
    print(f"  Round {r}: LLM Acc={acc_str}  F1={f1_str}  |  R(t)={sr}  GNN→LLM:{gnn_sel}(acc={gnn_pa_str})  LLM→GNN:{llm_sel}(acc={llm_pa_str})")

# Save summary
summary_path = os.path.join(run_dir, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nFull summary saved to: {summary_path}")
PYEOF
