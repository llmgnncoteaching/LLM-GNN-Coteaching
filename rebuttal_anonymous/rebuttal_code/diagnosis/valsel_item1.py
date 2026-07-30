# Extracted analysis (ran on server against results dirs).
import json, os, glob, random
random.seed(0)
def load(base):
    # returns list of (round, [(correct_bool, label)]) with fixed node order
    rounds={}
    for t in range(1,9):
        f=f"{base}/round{t}/test_predictions.jsonl"
        if not os.path.exists(f): continue
        rows=[]
        for line in open(f):
            d=json.loads(line)
            pred=d["predict"].strip().replace("<|eot_id|>","").strip()
            lab=d["label"].strip().replace("<|eot_id|>","").strip()
            rows.append((pred==lab, lab))
        rounds[t]=rows
    return rounds
def valsel(base, k, ndraws=500):
    R=load(base)
    ts=sorted(R)
    if not ts: return None
    n=len(R[ts[0]])
    labels=[R[ts[0]][i][1] for i in range(n)]
    # group indices by class
    from collections import defaultdict
    byc=defaultdict(list)
    for i,l in enumerate(labels): byc[l].append(i)
    corr={t:[R[t][i][0] for i in range(n)] for t in ts}
    best_round_acc=max(sum(corr[t])/n for t in ts)*100
    final_acc=sum(corr[ts[-1]])/n*100
    picks=[]; vsel=[]
    for _ in range(ndraws):
        val=set()
        for c,idxs in byc.items():
            if len(idxs)>=k: val.update(random.sample(idxs,k))
            else: val.update(idxs)
        rem=[i for i in range(n) if i not in val]
        # pick round by val acc
        bt=max(ts, key=lambda t: sum(corr[t][i] for i in val))
        picks.append(bt)
        vsel.append(sum(corr[bt][i] for i in rem)/len(rem)*100)
    import statistics as st
    from collections import Counter
    mode_round=Counter(picks).most_common(3)
    return dict(best=best_round_acc, final=final_acc,
                vsel_mean=st.mean(vsel), vsel_std=st.pstdev(vsel),
                picks=mode_round, rounds=ts)
for ds,base,k in [("cora","rtabl_cora_lin/results/co_teaching/cora_3shot_seed42",3),
                  ("pubmed","rtabl_pubmed_lin/results/co_teaching/pubmed_3shot_seed42",3),
                  ("arxiv","rtabl_arxiv_lin/results/co_teaching/arxiv_3shot_seed42",3)]:
    r=valsel(base,k)
    if r: print(f"{ds}: best={r['best']:.2f} valsel={r['vsel_mean']:.2f}+-{r['vsel_std']:.2f} final={r['final']:.2f} picks={r['picks']}")
    else: print(f"{ds}: no data")
