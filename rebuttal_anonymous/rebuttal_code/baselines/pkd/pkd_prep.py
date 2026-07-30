#!/usr/bin/env python
"""PKD baseline step 1: generate every input PKD needs, from our PyG-style .pt.

PKD (preference-based KD) distills from 4 GNN *teacher* logits into a student GNN,
with a PPO agent (per-node embedding -> pick 1 of 4 teachers). The public repo ships
none of the precomputed artifacts; this script produces all of them at k-shot:

  {ds}_feature.npy      (N, F) float32     node features x
  {ds}_edge_index.npy   (2, E) int64       edges
  {ds}_labels.npy       (N,)   int64        ground-truth y
  predicted_{ds}_labels_all.npy (N,) int64 LLM/SBERT pseudo-labels (all nodes)
  {ds}_pca.npy          (N, D) float32      PCA'd SBERT embeddings (RL agent state)
  teacher_logits.npy    (4, N, C) float32   the 4 GNN teachers (gcn/gat/appnp/h2gcn)
  {ds}_trueidx.npy      (k*C,) int64        the k-shot labeled train nodes
  {ds}_testidx.npy      (<=1000,) int64     held-out test nodes (from test_mask)
Also writes PKD-expected teacher layouts (logits/semi/*.pt and outputs/{ds}/{lr}/*.npy)
and stub modules (config.py, neighbor_select.py, get_paper_txt.py, data_amazon.py,
read.py) so the unmodified repo imports cleanly.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
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


# ------------------------------ teacher GNNs ------------------------------- #
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
    """Lightweight H2GCN-style: ego + 1-hop + 2-hop separated, then MLP-mix."""
    def __init__(self, fi, h, c):
        super().__init__()
        self.lin = nn.Linear(fi, h)
        self.out = nn.Linear(h * 3, c)

    def forward(self, x, ei, adj_norm):
        h0 = F.relu(self.lin(x))
        h1 = adj_norm @ h0
        h2 = adj_norm @ h1
        z = torch.cat([h0, h1, h2], dim=1)
        z = F.dropout(z, 0.5, self.training)
        return self.out(z)


def norm_adj(edge_index, n, device):
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp
    A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=n).tocsr()
    A = A + sp.eye(n)
    d = np.asarray(A.sum(1)).flatten()
    dinv = np.power(d, -0.5, where=d > 0); dinv[d == 0] = 0
    D = sp.diags(dinv)
    An = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([An.row, An.col]), dtype=torch.long, device=device)
    val = torch.tensor(An.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def train_teacher(kind, x, ei, y, tr, va, te, c, device, epochs=300):
    n, fi = x.size(0), x.size(1)
    adj = norm_adj(ei, n, device) if kind == "h2gcn" else None
    m = {"gcn": TeacherGCN, "gat": TeacherGAT, "appnp": TeacherAPPNP,
         "h2gcn": TeacherH2GCN}[kind](fi, 128, c).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    best_va, best_logits, best_te, patience = -1, None, 0, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out = m(x, ei, adj) if kind == "h2gcn" else m(x, ei)
        loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out = m(x, ei, adj) if kind == "h2gcn" else m(x, ei)
            pred = out.argmax(1)
            va_acc = (pred[va] == y[va]).float().mean().item() if va.numel() else 0.0
            te_acc = (pred[te] == y[te]).float().mean().item() if te.numel() else 0.0
        if va_acc >= best_va:
            best_va, best_te = va_acc, te_acc
            best_logits = out.detach().float().cpu().numpy()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                break
    print(f"  teacher {kind:6s}: val {best_va*100:5.2f}  test {best_te*100:5.2f}")
    return best_logits


# --------------------------------------------------------------------------- #
def sbert_encode(texts, name, device):
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(name, device=device)
    return sb.encode(list(texts), batch_size=256, show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float32)


def pseudo_labels_sbert(emb, y, trueidx, c):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(emb[trueidx], y[trueidx])
    return clf.predict(emb).astype(np.int64)


def pseudo_labels_llm(texts, label_names, endpoint, model):
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    cli = OpenAI(base_url=endpoint, api_key="EMPTY", timeout=120)
    labs = ", ".join(label_names)
    low = [l.lower() for l in label_names]

    def one(t):
        for _ in range(3):
            try:
                r = cli.chat.completions.create(model=model, max_tokens=16, temperature=0.0,
                    messages=[{"role": "system", "content": "Classify the text. Reply with exactly one class name."},
                              {"role": "user", "content": f"Classes: {labs}.\nText: {str(t)[:2000]}\nClass:"}])
                a = (r.choices[0].message.content or "").strip().lower()
                for j, ln in enumerate(low):
                    if ln in a or a in ln:
                        return j
                return 0
            except Exception:
                pass
        return 0
    from tqdm import tqdm
    out = [0] * len(texts)
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, r in enumerate(tqdm(ex.map(one, texts), total=len(texts), desc="llm-pl")):
            out[i] = r
    return np.array(out, dtype=np.int64)


STUBS = {
    "config.py": ('CUDA_VISIBLE_DEVICES = "0"\nUSE_TORCH = "1"\nCPU_NUMS = "4"\n'),
    "neighbor_select.py": ("def get_indices_list(*a, **k):\n    return None\n"),
    "get_paper_txt.py": ("def get_text(*a, **k):\n    return None, None\n"),
    "data_amazon.py": ("def get_data(*a, **k):\n    return None\n"),
    "read.py": ("def get_raw_text_webkb(*a, **k):\n    return None, None\n"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--outdir", required=True, help="datadir for pkd_main_generic + npy inputs")
    ap.add_argument("--pkd_dir", default="", help="PKD-main dir to also write repo-layout teacher files + stubs")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--label_rate", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_cap", type=int, default=1000)
    ap.add_argument("--pseudo_from", choices=["sbert", "llm"], default="sbert")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="llama3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset
    g = torch.load(args.pt, weights_only=False)
    x = g.x.float(); y = g.y.view(-1).long()
    ei = g.edge_index.long()
    n, fi = x.size(0), x.size(1); c = int(y.max()) + 1
    label_names = [str(l) for l in getattr(g, "label_name", [str(i) for i in range(c)])]
    print(f"[prep] {ds}: N={n} F={fi} C={c} E={ei.size(1)}")

    def mask(name):
        m = getattr(g, name, None)
        return m.cpu().bool() if torch.is_tensor(m) else None
    tr_m, va_m, te_m = mask("train_mask"), mask("val_mask"), mask("test_mask")

    trueidx = kshot_train_idx(y, tr_m, args.shots, args.seed)   # k-shot labeled train
    rng = np.random.RandomState(args.seed)
    test_pool = np.where(te_m.cpu().numpy())[0] if te_m is not None else \
        np.setdiff1d(np.arange(n), trueidx)
    testidx = np.sort(rng.permutation(test_pool)[:args.test_cap]).astype(np.int64)
    va_pool = np.where(va_m.cpu().numpy())[0] if va_m is not None else \
        np.setdiff1d(test_pool, testidx)[:1000]

    # ---- base npy inputs -------------------------------------------------- #
    np.save(f"{args.outdir}/{ds}_feature.npy", x.cpu().numpy().astype(np.float32))
    np.save(f"{args.outdir}/{ds}_edge_index.npy", ei.cpu().numpy().astype(np.int64))
    np.save(f"{args.outdir}/{ds}_labels.npy", y.cpu().numpy().astype(np.int64))
    np.save(f"{args.outdir}/{ds}_trueidx.npy", trueidx)
    np.save(f"{args.outdir}/{ds}_testidx.npy", testidx)

    # ---- 4 GNN teachers (trained at k-shot) ------------------------------- #
    xd, eid, yd = x.to(dev), ei.to(dev), y.to(dev)
    tr = torch.tensor(trueidx, device=dev)
    va = torch.tensor(va_pool, dtype=torch.long, device=dev)
    te = torch.tensor(testidx, device=dev)
    print("[prep] training 4 GNN teachers at %d-shot:" % args.shots)
    names = ["gcn", "gat", "appnp", "h2gcn"]
    tlogits = [train_teacher(k, xd, eid, yd, tr, va, te, c, dev) for k in names]
    teacher = np.stack(tlogits, axis=0).astype(np.float32)   # (4, N, C)
    np.save(f"{args.outdir}/teacher_logits.npy", teacher)

    # ---- SBERT + PCA embeddings (RL state) -------------------------------- #
    emb = sbert_encode(g.raw_texts, args.sbert, dev)
    from sklearn.decomposition import PCA
    d = min(args.pca_dim, emb.shape[1], emb.shape[0])
    pca = PCA(n_components=d, random_state=args.seed).fit(emb)
    pca_emb = pca.transform(emb).astype(np.float32)
    np.save(f"{args.outdir}/{ds}_pca.npy", pca_emb)

    # ---- pseudo-labels ---------------------------------------------------- #
    if args.pseudo_from == "sbert":
        pl = pseudo_labels_sbert(emb, y.cpu().numpy(), trueidx, c)
    else:
        pl = pseudo_labels_llm(list(g.raw_texts), label_names, args.endpoint, args.model)
    pl[trueidx] = y.cpu().numpy()[trueidx]                          # gold overrides on labeled
    np.save(f"{args.outdir}/predicted_{ds}_labels_all.npy", pl)
    print(f"[prep] pseudo-label ({args.pseudo_from}) train-node acc "
          f"{(pl[testidx]==y.cpu().numpy()[testidx]).mean()*100:.2f} on test")

    # ---- repo-layout copies + stubs (so unmodified PKD imports/loads) ----- #
    if args.pkd_dir:
        p = args.pkd_dir
        os.makedirs(f"{p}/logits/semi", exist_ok=True)
        os.makedirs(f"{p}/outputs/{ds}/{args.label_rate}", exist_ok=True)
        repo_names = {"gcn": names[0], "gat": names[1], "appnp": names[2], "h2gcn": names[3]}
        for i, nm in enumerate(names):
            torch.save(teacher[i], f"{p}/logits/semi/{nm}_{args.label_rate}_best.pt")
            np.save(f"{p}/outputs/{ds}/{args.label_rate}/{nm}_{ds}.npy", teacher[i])
        # extra aliases some datasets expect (gcn2)
        np.save(f"{p}/outputs/{ds}/{args.label_rate}/gcn2_{ds}.npy", teacher[0])
        for fn, body in STUBS.items():
            fp = f"{p}/{fn}"
            if not os.path.exists(fp):
                open(fp, "w").write(body)
        for sub in ("feature", "edge_index", "labels"):
            src = f"{args.outdir}/{ds}_{sub}.npy"
            if os.path.abspath(src) != os.path.abspath(f"{p}/{ds}_{sub}.npy"):
                np.save(f"{p}/{ds}_{sub}.npy", np.load(src))
        np.save(f"{p}/predicted_{ds}_labels_all.npy", pl)
        print(f"[prep] wrote repo-layout teachers + stubs into {p}")

    print(f"[prep] DONE {ds}: teacher{tuple(teacher.shape)} pca{tuple(pca_emb.shape)} "
          f"trueidx={len(trueidx)} testidx={len(testidx)}")


if __name__ == "__main__":
    main()
