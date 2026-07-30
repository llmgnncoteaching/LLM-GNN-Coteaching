"""
Faithful PKD prep: LLM-annotate nodes to EXPAND the teacher training set
(PKD's GNS/GNPS mechanism), then train the 4 teacher GNNs on gold+LLM labels.
Two stages so vLLM and torch-GNN training do not fight for GPU memory:
  --stage annotate : offline vLLM, direct prompt over ALL nodes -> pl_llm.npy
  --stage train    : load pl_llm.npy, build expanded train set, train teachers,
                     write teacher_logits.npy / *_pca.npy / predicted_*_labels_all.npy
                     / *_trueidx.npy / *_testidx.npy  (inputs for pkd_main_generic).
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kshot_train_idx(y, train_mask, k, seed):
    rng = np.random.RandomState(seed)
    y = y.cpu().numpy()
    pool = np.where(train_mask.cpu().numpy())[0] if train_mask is not None else np.arange(len(y))
    idx = []
    for c in range(int(y.max()) + 1):
        cand = pool[y[pool] == c]
        if len(cand):
            idx.extend(rng.permutation(cand)[:k].tolist())
    return np.array(sorted(idx), dtype=np.int64)


class TeacherGCN(nn.Module):
    def __init__(self, fi, h, c):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1, self.c2 = GCNConv(fi, h), GCNConv(h, c)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(x, ei)), 0.5, self.training)
        return self.c2(x, ei)


class TeacherGAT(nn.Module):
    def __init__(self, fi, h, c):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.c1 = GATConv(fi, h, heads=4, dropout=0.5)
        self.c2 = GATConv(h * 4, c, heads=1, concat=False, dropout=0.5)
    def forward(self, x, ei):
        x = F.elu(self.c1(x, ei))
        return self.c2(x, ei)


class TeacherAPPNP(nn.Module):
    def __init__(self, fi, h, c):
        super().__init__()
        from torch_geometric.nn import APPNP
        self.l1, self.l2 = nn.Linear(fi, h), nn.Linear(h, c)
        self.prop = APPNP(K=10, alpha=0.1)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.l1(F.dropout(x, 0.5, self.training))), 0.5, self.training)
        return self.prop(self.l2(x), ei)


class TeacherH2GCN(nn.Module):
    def __init__(self, fi, h, c):
        super().__init__()
        self.lin = nn.Linear(fi, h)
        self.out = nn.Linear(h * 3, c)
    def forward(self, x, ei, adj_norm):
        h0 = F.relu(self.lin(x)); h1 = adj_norm @ h0; h2 = adj_norm @ h1
        z = F.dropout(torch.cat([h0, h1, h2], dim=1), 0.5, self.training)
        return self.out(z)


def norm_adj(edge_index, n, device):
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp
    A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=n).tocsr() + sp.eye(n)
    d = np.asarray(A.sum(1)).flatten()
    dinv = np.power(d, -0.5, where=d > 0); dinv[d == 0] = 0
    D = sp.diags(dinv); An = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([An.row, An.col]), dtype=torch.long, device=device)
    val = torch.tensor(An.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def train_teacher(kind, x, ei, y_tr, tr, y_ev, va, te, c, device, epochs=300):
    """y_tr: labels used for TRAIN loss (gold+LLM). y_ev: GOLD labels for val/test eval."""
    n, fi = x.size(0), x.size(1)
    adj = norm_adj(ei, n, device) if kind == "h2gcn" else None
    m = {"gcn": TeacherGCN, "gat": TeacherGAT, "appnp": TeacherAPPNP,
         "h2gcn": TeacherH2GCN}[kind](fi, 128, c).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    best_va, best_logits, best_te, patience = -1, None, 0, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out = m(x, ei, adj) if kind == "h2gcn" else m(x, ei)
        loss = F.cross_entropy(out[tr], y_tr[tr]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out = m(x, ei, adj) if kind == "h2gcn" else m(x, ei)
            pred = out.argmax(1)
            va_acc = (pred[va] == y_ev[va]).float().mean().item() if va.numel() else 0.0
            te_acc = (pred[te] == y_ev[te]).float().mean().item() if te.numel() else 0.0
        if va_acc >= best_va:
            best_va, best_te = va_acc, te_acc
            best_logits = out.detach().float().cpu().numpy(); patience = 0
        else:
            patience += 1
            if patience > 50: break
    print(f"  teacher {kind:6s}: val {best_va*100:5.2f}  test {best_te*100:5.2f}", flush=True)
    return best_logits


def llm_annotate(texts, label_names, model_path, gpu_frac=0.6):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_frac,
              max_model_len=4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    labs = ", ".join(label_names); low = [l.lower() for l in label_names]
    def build(t):
        u = f"Classes: {labs}.\nText: {str(t)[:2000]}\nReply with exactly one class name from the list.\nClass:"
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    prompts = [build(t) for t in texts]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16))
    res = []
    for o in outs:
        a = o.outputs[0].text.strip().lower(); lab = 0
        for j, ln in enumerate(low):
            if ln in a: lab = j; break
        res.append(lab)
    return np.array(res, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pkd_dir", default="")
    ap.add_argument("--stage", choices=["annotate", "train"], required=True)
    ap.add_argument("--model_path", default="/project/anon/rebuttal/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--label_rate", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_cap", type=int, default=1000)
    ap.add_argument("--expand_frac", type=float, default=0.48, help="fraction of N used as expanded train set")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset
    g = torch.load(args.pt, weights_only=False)
    x = g.x.float(); y = g.y.view(-1).long(); ei = g.edge_index.long()
    n, fi = x.size(0), x.size(1); c = int(y.max()) + 1
    label_names = [str(l) for l in getattr(g, "label_name", [str(i) for i in range(c)])]
    print(f"[prep-faithful] {ds}: N={n} F={fi} C={c} stage={args.stage} expand={args.expand_frac}", flush=True)

    if args.stage == "annotate":
        pl = llm_annotate(list(g.raw_texts), label_names, args.model_path)
        acc = (pl == y.cpu().numpy()).mean() * 100
        np.save(f"{args.outdir}/pl_llm_{ds}.npy", pl)
        print(f"[annotate] LLM whole-graph label acc {acc:.2f}  saved pl_llm_{ds}.npy", flush=True)
        return

    # ---- stage train ----
    def mask(name):
        m = getattr(g, name, None)
        return m.cpu().bool() if torch.is_tensor(m) else None
    tr_m, va_m, te_m = mask("train_mask"), mask("val_mask"), mask("test_mask")
    trueidx = kshot_train_idx(y, tr_m, args.shots, args.seed)
    rng = np.random.RandomState(args.seed)
    test_pool = np.where(te_m.cpu().numpy())[0] if te_m is not None else np.setdiff1d(np.arange(n), trueidx)
    testidx = np.sort(rng.permutation(test_pool)[:args.test_cap]).astype(np.int64)
    va_pool = (np.where(va_m.cpu().numpy())[0] if va_m is not None else
               np.setdiff1d(test_pool, testidx)[:1000])
    pl = np.load(f"{args.outdir}/pl_llm_{ds}.npy")

    # ---- EXPANSION: gold on k-shot + LLM labels on sampled non-test/non-val nodes ---
    ynp = y.cpu().numpy()
    excl = set(testidx.tolist()) | set(np.asarray(va_pool).tolist()) | set(trueidx.tolist())
    pool = np.array([i for i in range(n) if i not in excl], dtype=np.int64)
    n_expand = max(0, int(args.expand_frac * n) - len(trueidx))
    sel = np.sort(rng.permutation(pool)[:n_expand]).astype(np.int64)
    y_train = ynp.copy()
    y_train[sel] = pl[sel]                      # LLM labels on expanded set
    tr_exp = np.sort(np.concatenate([trueidx, sel])).astype(np.int64)
    print(f"[train] expanded train: {len(trueidx)} gold + {len(sel)} LLM = {len(tr_exp)} "
          f"({100*len(tr_exp)/n:.1f}% of N); LLM label acc on expanded {(pl[sel]==ynp[sel]).mean()*100 if len(sel) else 0:.2f}", flush=True)

    xd, eid = x.to(dev), ei.to(dev)
    y_tr = torch.tensor(y_train, device=dev); y_ev = torch.tensor(ynp, device=dev)
    tr = torch.tensor(tr_exp, device=dev); va = torch.tensor(np.asarray(va_pool), dtype=torch.long, device=dev)
    te = torch.tensor(testidx, device=dev)
    names = ["gcn", "gat", "appnp", "h2gcn"]
    tlogits = [train_teacher(k, xd, eid, y_tr, tr, y_ev, va, te, c, dev) for k in names]
    teacher = np.stack(tlogits, axis=0).astype(np.float32)

    np.save(f"{args.outdir}/{ds}_feature.npy", x.cpu().numpy().astype(np.float32))
    np.save(f"{args.outdir}/{ds}_edge_index.npy", ei.cpu().numpy().astype(np.int64))
    np.save(f"{args.outdir}/{ds}_labels.npy", ynp.astype(np.int64))
    np.save(f"{args.outdir}/{ds}_trueidx.npy", trueidx)
    np.save(f"{args.outdir}/{ds}_testidx.npy", testidx)
    np.save(f"{args.outdir}/teacher_logits.npy", teacher)

    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(args.sbert, device=dev)
    emb = sb.encode(list(g.raw_texts), batch_size=256, convert_to_numpy=True).astype(np.float32)
    from sklearn.decomposition import PCA
    d = min(args.pca_dim, emb.shape[1], emb.shape[0])
    pca_emb = PCA(n_components=d, random_state=args.seed).fit_transform(emb).astype(np.float32)
    np.save(f"{args.outdir}/{ds}_pca.npy", pca_emb)

    pl2 = pl.copy(); pl2[trueidx] = ynp[trueidx]
    np.save(f"{args.outdir}/predicted_{ds}_labels_all.npy", pl2)

    if args.pkd_dir:
        p = args.pkd_dir
        os.makedirs(f"{p}/logits/semi", exist_ok=True)
        os.makedirs(f"{p}/outputs/{ds}/{args.label_rate}", exist_ok=True)
        for i, nm in enumerate(names):
            torch.save(teacher[i], f"{p}/logits/semi/{nm}_{args.label_rate}_best.pt")
            np.save(f"{p}/outputs/{ds}/{args.label_rate}/{nm}_{ds}.npy", teacher[i])
        np.save(f"{p}/outputs/{ds}/{args.label_rate}/gcn2_{ds}.npy", teacher[0])
    print(f"[train] teachers on expanded set DONE for {ds}", flush=True)


if __name__ == "__main__":
    main()
