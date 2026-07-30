# Extracted analysis (ran on server against results dirs).
import csv, json, os, glob
base="rtabl_arxiv_f10b/results/co_teaching/arxiv_3shot_seed42"
f=glob.glob(base+"/progress.csv")[0]
r=[x for x in csv.reader(open(f)) if x and x[0].isdigit()]
print("f10b GNN_max","%.2f"%max(float(x[1]) for x in r),"LLM_max","%.2f"%(max(float(x[3]) for x in r)*100),"rounds",len(r))
print("f10b LLM per round:", [round(float(x[3])*100,1) for x in r])
print("f10b GNN_pseudo per round:", [round(float(x[10]),3) for x in r])
# error jaccard on f10b
def load_errs(bd):
    per={}
    for t in range(1,9):
        pf=f"{bd}/round{t}/test_predictions.jsonl"
        if not os.path.exists(pf): continue
        errs=set()
        for i,line in enumerate(open(pf)):
            d=json.loads(line)
            if d["predict"].strip().replace("<|eot_id|>","").strip()!=d["label"].strip().replace("<|eot_id|>","").strip(): errs.add(i)
        per[t]=errs
    return per
per=load_errs(base)
ts=sorted(per)
def jac(a,b): u=len(a|b); return len(a&b)/u if u else 0
print("f10b error-set jac(t,t+1):", [round(jac(per[ts[j]],per[ts[j+1]]),3) for j in range(len(ts)-1)])
if ts: 
    e1,eL=per[ts[0]],per[ts[-1]]
    print("f10b jac(r1,rL)=%.3f persistence=%.3f nerr r1=%d rL=%d"%(jac(e1,eL), len(e1&eL)/len(e1) if e1 else 0, len(e1), len(eL)))
