#!/bin/bash
D=$1; G=$2
cd /home/anon/graph_token_bench/upstream/LLaGA
export PATH=/home/anon/anaconda3/envs/llaga38/bin:$PATH; source ~/anaconda3/etc/profile.d/conda.sh; conda activate llaga38
export PYTHONPATH=$PWD:$PYTHONPATH WANDB_DISABLED=true OMP_NUM_THREADS=6
PY=/home/anon/anaconda3/envs/llaga38/bin/python
CK=./checkpoints/${D}/llaga-qwen3b-mpnet-nc
ANS=/home/anon/graph-token-reasoning/outputs/llagaqwen_${D}_ans.jsonl; rm -f $ANS
R=/home/anon/graph-token-reasoning/outputs/LLAGAQWEN_${D}.txt
echo "=== EVAL START $(date) D=$D ===" >> $R
CUDA_VISIBLE_DEVICES=$G $PY eval/eval_pretrain_qwen.py --model_path $CK --conv_mode mpt --dataset $D --pretrained_embedding_type mpnet --use_hop 2 --sample_neighbor_size 10 --answers_file $ANS --task nc --cache_dir ../../checkpoint --template ND >> $R 2>&1
$PY - "$ANS" "$D" >> $R 2>&1 <<'PYS'
import json,sys,re
ans,D=sys.argv[1],sys.argv[2]
cats=[c.strip() for c in json.load(open(f"dataset/{D}/processed_data.pt")) ] if False else None
ok=t=0
rows=[json.loads(l) for l in open(ans)]
for d in rows:
    pred=d['text'].strip().lower(); gt=d['gt'].strip().lower()
    t+=1
    ok+= (gt==pred or pred.startswith(gt) or gt in pred)
print(f"LLAGAQWEN {D} NC acc = {ok}/{t} = {100*ok/max(1,t):.1f}%")
PYS
echo "DONE_EVAL_${D}" >> $R
