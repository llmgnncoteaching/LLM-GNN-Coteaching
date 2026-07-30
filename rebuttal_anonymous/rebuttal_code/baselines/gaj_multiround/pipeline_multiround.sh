#!/bin/bash
# ============================================================================
# Multi-round (iterative) GNN-as-Judge  —  compute-matched baseline for the
# NeurIPS rebuttal (item 2). GAJ is single-pass; here we iterate its native
# cycle [ select-fixed -> LLM infer -> GNN-judge -> DPO ] for T rounds so GAJ
# gets a Co-Teaching-equivalent number of LLM update rounds. GNN and the SFT
# warm start are trained ONCE; the DPO adapter accumulates across rounds.
# Per-round test accuracy is logged to progress.csv for the trajectory.
# ============================================================================
set -uo pipefail
if [ -f "config.sh" ]; then source config.sh; fi

DATASET=${1:-"cora"}
SHOT_COUNT=${2:-"3"}
SEED=${3:-"42"}
NUM_ROUNDS=${4:-"10"}
TAG=${EXP_TAG:-"gajmr"}

export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES:-"3"}
export HF_HOME="$WORKSPACE_DIR/huggingface_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
source "$CONDA_SH"; conda activate "$ENV_NAME"

MAIN_EXP_DIR="$WORKSPACE_DIR/results/gnn_as_judge_multiround"
DATASET_DIR="$LF_DIR/data"; DATASET_INFO_FILE="$DATASET_DIR/dataset_info.json"
RUN_ID="${DATASET}_${SHOT_COUNT}shot_seed${SEED}_${TAG}"
RUN_DIR="$MAIN_EXP_DIR/$RUN_ID"
mkdir -p "$HF_HOME" "$DATASET_DIR" "$RUN_DIR"
SFT_DATASET_PREFIX="${DATASET}_sft_${SHOT_COUNT}_shot"
PROGRESS="$RUN_DIR/progress.csv"
GNN_MODEL_PATH="$PROJECT_DIR/results/GNN/${DATASET}_${SHOT_COUNT}_shot_best_model_run0.pt"

echo "=== Multi-round GAJ: $RUN_ID | rounds=$NUM_ROUNDS | GPU=$CUDA_VISIBLE_DEVICES ==="

# ---------- STAGE 0 (once): train GNN judge ----------
if [ ! -f "$GNN_MODEL_PATH" ]; then
  echo "--- Stage 0: train GNN judge ---"
  mkdir -p "$PROJECT_DIR/results/GNN"
  cd "$PROJECT_DIR/GNN"
  python main.py --dataset "$DATASET" --shots "$SHOT_COUNT" --gnn_type "$GNN_TYPE" \
    --hidden_dim "$GNN_HIDDEN_DIM" --n_layers "$GNN_LAYERS" --epochs 500 \
    --seed "$SEED" --device "cuda:0" || { echo "GNN train failed"; exit 1; }
fi

# ---------- STAGE 1 (once): SFT data ----------
echo "--- Stage 1: create SFT data ---"
cd "$PROJECT_DIR"
python create_sft.py --dataset "$DATASET" --output "$DATASET_DIR/${DATASET}_sft.json" \
  --shots "$SHOT_COUNT" --seed "$SEED" --path_prefix "." || exit 1
python - <<EOF
import json, os
p="$DATASET_INFO_FILE"; prefix="$SFT_DATASET_PREFIX"
info=json.load(open(p)) if os.path.exists(p) else {}
for s in ["train","val","test","unlabeled"]:
    info[f"{prefix}_{s}"]={"file_name":f"{prefix}_{s}.json","formatting":"sharegpt","columns":{"messages":"conversations"}}
json.dump(info,open(p,'w'),indent=2,ensure_ascii=False)
EOF

# ---------- STAGE 2 (once): SFT warm start ----------
SFT_OUTDIR="$RUN_DIR/sft/model"; mkdir -p "$SFT_OUTDIR"
echo "--- Stage 2: SFT warm start ---"
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES llamafactory-cli train --stage sft --do_train \
  --model_name_or_path "$BASE_MODEL_PATH" --dataset_dir "$DATASET_DIR" \
  --dataset "${SFT_DATASET_PREFIX}_train" --template "$TEMPLATE" \
  --finetuning_type lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_target all \
  --output_dir "$SFT_OUTDIR" --overwrite_cache --overwrite_output_dir \
  --cutoff_len 2048 --preprocessing_num_workers 16 \
  --per_device_train_batch_size $BATCH_SIZE_SFT --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --lr_scheduler_type cosine --logging_steps 20 --save_steps 500 \
  --learning_rate $LEARNING_RATE_SFT --num_train_epochs $EPOCHS_SFT \
  --plot_loss --bf16 --save_total_limit 1 2>&1 | tee "$RUN_DIR/sft_train.log"
CUR_ADAPTER="$SFT_OUTDIR"
LC=$(ls -dt "$SFT_OUTDIR"/checkpoint-* 2>/dev/null | head -1); [ -n "$LC" ] && CUR_ADAPTER="$LC"

# ---------- STAGE 3 (once): select influential nodes + filter ----------
echo "--- Stage 3: select influential nodes ---"
SELECTED_NODES_FILE="$RUN_DIR/${RUN_ID}_selected_nodes.json"
cd "$PROJECT_DIR"
python select_influential_nodes.py --dataset "$DATASET" --k $TOPK_INFLUENTIAL \
  --output_file "$SELECTED_NODES_FILE" --shots "$SHOT_COUNT" --seed "$SEED" \
  --method auto --max_subgraph_nodes $MAX_SUBGRAPH_NODES --max_distance 3 --path_prefix "." || exit 1
SELECTED_DATASET_NAME="${SFT_DATASET_PREFIX}_selected_${TAG}"
python - <<EOF
import json, os
sel=set(json.load(open("$SELECTED_NODES_FILE"))['selected_node_ids'])
allids=json.load(open("$DATASET_DIR/${DATASET}_${SHOT_COUNT}_shot_unlabeled_node_ids.json"))['selected_node_ids']
unl=json.load(open("$DATASET_DIR/${SFT_DATASET_PREFIX}_unlabeled.json"))
filt=[];order=[]
for i,nid in enumerate(allids):
    if nid in sel: filt.append(unl[i]); order.append(nid)
json.dump(filt,open("$DATASET_DIR/${SELECTED_DATASET_NAME}.json",'w'),ensure_ascii=False,indent=2)
json.dump({"selected_node_ids":order},open("${SELECTED_NODES_FILE%.json}_ordered.json",'w'),indent=2)
info=json.load(open("$DATASET_INFO_FILE"))
info["$SELECTED_DATASET_NAME"]={"file_name":"${SELECTED_DATASET_NAME}.json","formatting":"sharegpt","columns":{"messages":"conversations"}}
json.dump(info,open("$DATASET_INFO_FILE",'w'),indent=2,ensure_ascii=False)
print(f"selected {len(filt)} nodes")
EOF

echo "round,llm_test_acc,llm_test_f1,n_dpo_pairs" > "$PROGRESS"

# ---------- ROUNDS: infer -> GNN-judge -> DPO -> eval ----------
for ((r=1; r<=NUM_ROUNDS; r++)); do
  echo "=================  ROUND $r / $NUM_ROUNDS  ================="
  RD="$RUN_DIR/round${r}"; mkdir -p "$RD"
  LLM_PRED="$RD/llm_preds.jsonl"
  DPO_NAME="${RUN_ID}_round${r}_dpo"
  DPO_JSON="$DATASET_DIR/${DPO_NAME}.json"; SFT_DPO_JSON="$DATASET_DIR/${DPO_NAME}_sft.json"

  # infer with current adapter on the fixed selected set
  cd "$PROJECT_DIR"
  VLLM_LOGGING_LEVEL=ERROR CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES python /home/anon/rebuttal_gaj/vllm_infer.py \
    --model_name_or_path "$BASE_MODEL_PATH" --adapter_name_or_path "$CUR_ADAPTER" \
    --dataset "$SELECTED_DATASET_NAME" --template "$TEMPLATE" \
    --dataset_dir "$DATASET_DIR" --save_name "$LLM_PRED" || { echo "infer failed r$r"; break; }

  # GNN judges -> preference pairs
  cd "$PROJECT_DIR"
  python create_wsft.py --dataset "$DATASET" \
    --selected_nodes_path "${SELECTED_NODES_FILE%.json}_ordered.json" \
    --pretrained_model "$GNN_MODEL_PATH" --llm_predictions "$LLM_PRED" \
    --dpo_output_path "$DPO_JSON" --sft_output_path "$SFT_DPO_JSON" \
    --confidence_threshold $CONFIDENCE_THRESHOLD --shots "$SHOT_COUNT" \
    --gnn_type "$GNN_TYPE" --hidden_dim $GNN_HIDDEN_DIM --n_layers $GNN_LAYERS \
    --seed "$SEED" --device "cuda:0" || { echo "judge failed r$r"; break; }
  NPAIRS=$(python -c "import json;print(len(json.load(open('$DPO_JSON'))))" 2>/dev/null || echo 0)
  python - <<EOF
import json,os
p="$DATASET_INFO_FILE"; ds="$DPO_NAME"
info=json.load(open(p)) if os.path.exists(p) else {}
info[ds]={"file_name":ds+".json","formatting":"sharegpt","ranking":True,"columns":{"messages":"conversations","chosen":"chosen","rejected":"rejected"}}
json.dump(info,open(p,'w'),indent=2,ensure_ascii=False)
EOF

  # DPO from current adapter -> new adapter for this round
  DPO_OUT="$RD/dpo/model"; mkdir -p "$DPO_OUT"
  cd "$PROJECT_DIR"
  CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES llamafactory-cli train --stage dpo --do_train \
    --model_name_or_path "$BASE_MODEL_PATH" --adapter_name_or_path "$CUR_ADAPTER" --create_new_adapter \
    --dataset_dir "$DATASET_DIR" --dataset "$DPO_NAME" --template "$TEMPLATE" \
    --finetuning_type lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_target all \
    --pref_beta $DPO_BETA --pref_loss orpo \
    --output_dir "$DPO_OUT" --overwrite_cache --overwrite_output_dir \
    --cutoff_len 2048 --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_DPO --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --lr_scheduler_type cosine --logging_steps 20 --save_steps 100 \
    --learning_rate $LEARNING_RATE_DPO --num_train_epochs $EPOCHS_DPO \
    --plot_loss --bf16 --save_total_limit 1 2>&1 | tee "$RD/dpo_train.log" || { echo "dpo failed r$r"; break; }
  NEW_ADAPTER="$DPO_OUT"; LC=$(ls -dt "$DPO_OUT"/checkpoint-* 2>/dev/null | head -1); [ -n "$LC" ] && NEW_ADAPTER="$LC"
  CUR_ADAPTER="$NEW_ADAPTER"

  # eval on test
  TEST_PRED="$RD/test_predictions.jsonl"
  cd "$PROJECT_DIR"
  VLLM_LOGGING_LEVEL=ERROR CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES python /home/anon/rebuttal_gaj/vllm_infer.py \
    --model_name_or_path "$BASE_MODEL_PATH" --adapter_name_or_path "$CUR_ADAPTER" \
    --dataset "${SFT_DATASET_PREFIX}_test" --template "$TEMPLATE" \
    --dataset_dir "$DATASET_DIR" --save_name "$TEST_PRED" || { echo "eval infer failed r$r"; break; }
  cd "$PROJECT_DIR"
  python evaluate_predictions.py --dataset "$DATASET" --pred_file "$TEST_PRED" \
    --output_dir "$RD/eval" --model_name "round${r}" --path_prefix "." || true
  python - <<EOF
import json,os
m=os.path.join("$RD","eval/round${r}/metrics.json")
acc=f1=""
if os.path.exists(m):
    d=json.load(open(m)); acc=d.get("accuracy",""); f1=d.get("macro_f1","")
open("$PROGRESS","a").write(f"$r,{acc},{f1},$NPAIRS\n")
print(f"[ROUND $r] test acc={acc} f1={f1} dpo_pairs=$NPAIRS")
EOF
done
echo "=== DONE. progress: $PROGRESS ==="
cat "$PROGRESS"
