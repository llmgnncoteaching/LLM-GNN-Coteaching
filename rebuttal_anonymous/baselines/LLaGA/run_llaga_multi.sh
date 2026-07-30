cd /home/anon/graph_token_bench/upstream/LLaGA
export PYTHONPATH=$PWD:$PYTHONPATH; export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
PY=/home/anon/anaconda3/envs/agentgl-gt/bin/python; D=$1; G=$2
if [ ! -f checkpoints/$D/llaga-mpnet-nc/mm_projector.bin ]; then
  CUDA_VISIBLE_DEVICES=$G $PY train/train_mem.py --model_name_or_path lmsys/vicuna-7b-v1.5-16k --version v1 --cache_dir ../../checkpoint \
    --pretrained_embedding_type mpnet --tune_mm_mlp_adapter True --mm_use_graph_start_end False --mm_use_graph_patch_token False --bf16 True \
    --output_dir ./checkpoints/$D/llaga-mpnet-nc --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --evaluation_strategy no --save_strategy epoch --learning_rate 2e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 20 --tf32 True --model_max_length 4096 --gradient_checkpointing True --lazy_preprocess True --report_to none \
    --use_hop 2 --sample_neighbor_size 10 --mm_projector_type linear --use_task nc --use_dataset $D --template ND > train_node_${D}.log 2>&1
fi
[ -f checkpoints/$D/llaga-mpnet-nc/mm_projector.bin ] || { echo "$D TRAIN_FAIL" >> llaga_node_results.txt; exit 1; }
CUDA_VISIBLE_DEVICES=$G $PY eval/eval_pretrain.py --model_path ./checkpoints/$D/llaga-mpnet-nc --model_base lmsys/vicuna-7b-v1.5-16k --conv_mode v1 \
  --dataset $D --pretrained_embedding_type mpnet --use_hop 2 --sample_neighbor_size 10 --template ND --task nc --answers_file eval_out/${D}_node.jsonl > eval_node_${D}.log 2>&1
$PY -c "import json;r=[json.loads(l) for l in open('eval_out/${D}_node.jsonl')];c=sum(1 for x in r if str(x.get('gt','')).split('(')[0].lower() in str(x.get('text','')).lower());print(f'LLaGA_mpnet ${D}: {100*c/len(r):.1f}%')" >> llaga_node_results.txt 2>&1
