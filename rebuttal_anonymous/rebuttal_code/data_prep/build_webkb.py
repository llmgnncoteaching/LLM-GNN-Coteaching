import torch, os, csv, ast, random
from torch_geometric.data import Data
os.chdir("/home/anon/rebuttal_gaj/webkb_dl")
random.seed(42); torch.manual_seed(42)
def build(ds, D):
    x = torch.load(f"{ds}_sbert_x.pt", weights_only=False).float()
    with open(f"{ds}_{D}.csv") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["node_id"]))
    N = len(rows)
    assert x.shape[0]==N, (x.shape, N)
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    raw_texts = [r["raw_text"] for r in rows]
    # label_name: index->category
    m={}
    for r in rows: m[int(r["label"])]=r["category"].strip()
    label_name=[m[i] for i in range(max(m)+1)]
    # edges from neighbor_ids
    src=[]; dst=[]
    for r in rows:
        i=int(r["node_id"])
        nbs=ast.literal_eval(r["neighbor_ids"]) if r["neighbor_ids"].strip() else []
        for j in nbs:
            src+=[i,int(j)]; dst+=[int(j),i]   # undirected
    edge_index=torch.tensor([src,dst],dtype=torch.long) if src else torch.zeros((2,0),dtype=torch.long)
    # masks 60/10/30
    idx=list(range(N)); random.shuffle(idx)
    ntr=int(0.6*N); nval=int(0.1*N)
    tr=set(idx[:ntr]); va=set(idx[ntr:ntr+nval]); te=set(idx[ntr+nval:])
    def mk(s): 
        mm=torch.zeros(N,dtype=torch.bool); 
        for k in s: mm[k]=True
        return mm
    d=Data(x=x,y=y,edge_index=edge_index)
    d.raw_texts=raw_texts; d.label_name=label_name
    d.train_mask=mk(tr); d.val_mask=mk(va); d.test_mask=mk(te)
    out=f"/home/anon/rebuttal_gaj/datasets/{ds}.pt"
    torch.save(d,out)
    print(f"{ds}: N={N} classes={len(label_name)} {label_name} edges={edge_index.shape[1]} x={tuple(x.shape)} train/val/test={len(tr)}/{len(va)}/{len(te)} -> {out}")
build("cornell","Cornell")
build("wisconsin","Wisconsin")
print("BUILD_DONE")
