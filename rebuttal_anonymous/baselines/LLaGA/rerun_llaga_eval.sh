cd /home/anon/graph_token_bench/upstream/LLaGA
export PYTHONPATH=$PWD:$PYTHONPATH; export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
PY=/home/anon/anaconda3/envs/agentgl-gt/bin/python
# retry on a free-ish GPU: wait until some GPU has >20GB free
for t in $(seq 1 60); do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '$2>20000{print $1; exit}')
  [ -n "$G" ] && break; sleep 60
done
G=${G:-0}
echo "eval on GPU $G $(date)" > llaga_eval_status.txt
CUDA_VISIBLE_DEVICES=$G $PY eval/eval_pretrain.py --model_path ./checkpoints/ogbn-arxiv/llaga-mpnet-nc \
  --model_base lmsys/vicuna-7b-v1.5-16k --conv_mode v1 --dataset arxiv --pretrained_embedding_type mpnet \
  --use_hop 2 --sample_neighbor_size 10 --template ND --task nc --answers_file eval_out/llaga_mpnet_preds.jsonl > eval_mpnet2.log 2>&1
echo "DONE $(date)" >> llaga_eval_status.txt
# compute accuracy from preds
$PY - <<'PYIN'
import json
try:
    rows=[json.loads(l) for l in open("eval_out/llaga_mpnet_preds.jsonl")]
    # LLaGA pred format: compare gpt answer vs gold
    c=0
    for r in rows:
        pred=str(r.get("text", r.get("pred",""))); gold=str(r.get("gt", r.get("label","")))
        if gold and gold.split("(")[0] in pred: c+=1
    print(f"LLAGA_MPNET_ARXIV: {c}/{len(rows)} = {100*c/max(len(rows),1):.1f}%")
except Exception as e: print("acc calc err:", e)
PYIN
