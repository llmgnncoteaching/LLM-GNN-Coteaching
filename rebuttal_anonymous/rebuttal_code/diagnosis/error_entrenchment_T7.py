# Extracted analysis (ran on server against results dirs).
import json, os
def load_errs(base, rounds=range(1,9)):
    per={}
    for t in rounds:
        f=f"{base}/round{t}/test_predictions.jsonl"
        if not os.path.exists(f): continue
        errs=set(); labels=[]
        for i,line in enumerate(open(f)):
            d=json.loads(line)
            pred=d["predict"].strip().replace("<|eot_id|>","").strip()
            lab=d["label"].strip().replace("<|eot_id|>","").strip()
            labels.append(lab)
            if pred!=lab: errs.add(i)
        per[t]=(errs,tuple(labels))
    return per
def jac(a,b): 
    u=len(a|b); return (len(a&b)/u) if u else 0.0
for name,base in [("CO-TEACH","rtabl_arxiv_lin/results/co_teaching/arxiv_3shot_seed42"),
                  ("SELF-TRAIN","selftrain_arxiv/results/co_teaching/arxiv_3shot_seed42")]:
    per=load_errs(base)
    if not per: print(name,"NO DATA"); continue
    ts=sorted(per)
    # verify same test order across rounds (labels tuple identical)
    labs=set(per[t][1] for t in ts)
    same = len(labs)==1
    print(f"\n=== {name} (rounds {ts}, test-order-consistent={same}) ===")
    print("round  n_err  jac(t,t+1)")
    for j,t in enumerate(ts):
        e=per[t][0]
        jn = jac(e, per[ts[j+1]][0]) if j+1<len(ts) else float('nan')
        print(f"  {t}    {len(e):4d}   {jn:.3f}")
    e1=per[ts[0]][0]; eL=per[ts[-1]][0]
    persist = len(e1&eL)/len(e1) if e1 else 0
    print(f"  jaccard(r{ts[0]},r{ts[-1]})={jac(e1,eL):.3f}  persistence(r1 errs still wrong r{ts[-1]})={persist:.3f}")
    print(f"  err_size r{ts[0]}={len(e1)} -> r{ts[-1]}={len(eL)}")
