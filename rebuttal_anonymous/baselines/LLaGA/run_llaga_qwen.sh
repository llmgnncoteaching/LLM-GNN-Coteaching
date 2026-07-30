#!/bin/bash
D=$1; G=$2
cd /home/anon/graph_token_bench/upstream/LLaGA
export PATH=/home/anon/anaconda3/envs/llaga38/bin:$PATH; source ~/anaconda3/etc/profile.d/conda.sh; conda activate llaga38
export DS_SKIP_CUDA_CHECK=1 CUDA_HOME=/usr/local/cuda
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$((29600+$G))
export PYTHONPATH=$PWD:$PYTHONPATH WANDB_DISABLED=true OMP_NUM_THREADS=6
PY=/home/anon/anaconda3/envs/llaga38/bin/python
OUT=./checkpoints/${D}/llaga-qwen3b-mpnet-nc
R=/home/anon/graph-token-reasoning/outputs/LLAGAQWEN_${D}.txt
echo "=== START $(date) LLaGA-Qwen3B D=$D GPU=$G ===" > $R
CUDA_VISIBLE_DEVICES=$G $PY train/train_mem_qwen.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct --version mpt --cache_dir ../../checkpoint \
  --pretrained_embedding_type mpnet --tune_mm_mlp_adapter False \
  --mm_use_graph_start_end False --mm_use_graph_patch_token False --bf16 True \
  --output_dir $OUT --num_train_epochs 1 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --deepspeed scripts/zero2_offload.json --max_grad_norm 1.0 --save_strategy no --learning_rate 2e-5 \
  --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 5 --tf32 True --model_max_length 4096 --gradient_checkpointing True \
  --lazy_preprocess True --report_to none --use_hop 2 --sample_neighbor_size 10 \
  --mm_projector_type 2-layer-mlp --use_task nc --use_dataset $D --template ND >> $R 2>&1
echo "DONE_LLAGAQWEN_${D} rc=$?" >> $R
