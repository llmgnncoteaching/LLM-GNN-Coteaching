cd /home/anon/graph_token_bench/upstream/LLaGA
export PYTHONPATH=$PWD:$PYTHONPATH
PY=/home/anon/anaconda3/envs/agentgl-gt/bin/python
ST=run_llaga_mpnet_status.txt; echo "TRAIN_START $(date)" > $ST
CUDA_VISIBLE_DEVICES=0 $PY train/train_mem.py \
  --model_name_or_path lmsys/vicuna-7b-v1.5-16k --version v1 --cache_dir ../../checkpoint \
  --pretrained_embedding_type mpnet --tune_mm_mlp_adapter True \
  --mm_use_graph_start_end False --mm_use_graph_patch_token False --bf16 True \
  --output_dir ./checkpoints/ogbn-arxiv/llaga-mpnet-nc --num_train_epochs 1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --evaluation_strategy no --save_strategy epoch --learning_rate 2e-3 --weight_decay 0. \
  --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 10 --tf32 True \
  --model_max_length 4096 --gradient_checkpointing True --lazy_preprocess True --report_to none \
  --use_hop 2 --sample_neighbor_size 10 --mm_projector_type linear --use_task nc --use_dataset arxiv --template ND > train_mpnet.log 2>&1
[ -d ./checkpoints/ogbn-arxiv/llaga-mpnet-nc ] || { echo "TRAIN_FAILED" >> $ST; exit 1; }
echo "TRAIN_DONE eval $(date)" >> $ST
CUDA_VISIBLE_DEVICES=0 $PY eval/eval_pretrain.py --model_path ./checkpoints/ogbn-arxiv/llaga-mpnet-nc \
  --model_base lmsys/vicuna-7b-v1.5-16k --conv_mode v1 --dataset arxiv --pretrained_embedding_type mpnet \
  --use_hop 2 --sample_neighbor_size 10 --template ND --task nc --answers_file llaga_mpnet_preds.jsonl > eval_mpnet.log 2>&1
echo "EVAL_DONE $(date)" >> $ST; grep -iE "acc|accuracy" eval_mpnet.log | tail -3 >> $ST
