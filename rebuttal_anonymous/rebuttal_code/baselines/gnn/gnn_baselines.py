"""Classical GNN baselines (GCN/GAT/SAGE) at k-shot on standard test split.
Prints: GNNBASE <dataset> <gnn> <shots> <acc>."""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def kshot_idx(y, tm, k, seed):
    rng = np.random.RandomState(seed); yn = y.cpu().numpy()
    pool = np.where(tm.cpu().numpy())[0] if tm is not None else np.arange(len(yn))
    idx = []
    for c in range(int(yn.max()) + 1):
        cand = pool[yn[pool] == c]
        if len(cand): idx.extend(rng.permutation(cand)[:k].tolist())
    return np.array(sorted(idx), dtype=np.int64)


class GNN(nn.Module):
    def __init__(self, fi, h, c, kind):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        self.kind = kind
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(fi, h), GCNConv(h, c)
        elif kind == "gat":
            self.c1 = GATConv(fi, h, heads=4, dropout=0.5)
            self.c2 = GATConv(h * 4, c, heads=1, concat=False, dropout=0.5)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(fi, h), SAGEConv(h, c)

    def forward(self, x, ei):
        if self.kind == "gat":
            x = F.elu(self.c1(x, ei)); return self.c2(x, ei)
        x = F.dropout(F.relu(self.c1(x, ei)), 0.5, self.training)
        return self.c2(x, ei)


def run(x, ei, y, tr, va, te, c, kind, dev, epochs=300):
    m = GNN(x.size(1), 128, c, kind).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    best_va, best_te, pat = -1, 0, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out = m(x, ei); loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(x, ei).argmax(1)
            va_a = (p[va] == y[va]).float().mean().item() if va.numel() else 0.0
            te_a = (p[te] == y[te]).float().mean().item() if te.numel() else 0.0
        if va_a >= best_va: best_va, best_te, pat = va_a, te_a, 0
        else:
            pat += 1
            if pat > 50: break
    return best_te * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--shots", type=int, nargs="+", required=True)
    ap.add_argument("--gnns", nargs="+", default=["gcn", "gat", "sage"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_cap", type=int, default=1000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.load(args.pt, weights_only=False)
    x = g.x.float().to(dev); ei = g.edge_index.long().to(dev); y = g.y.view(-1).long().to(dev)
    c = int(y.max()) + 1
    def mask(nm):
        m = getattr(g, nm, None); return m.cpu().bool() if torch.is_tensor(m) else None
    tm, vm, em = mask("train_mask"), mask("val_mask"), mask("test_mask")
    rng = np.random.RandomState(args.seed)
    tp = np.where(em.cpu().numpy())[0] if em is not None else np.arange(x.size(0))
    te = torch.tensor(np.sort(rng.permutation(tp)[:args.test_cap]), dtype=torch.long, device=dev)
    vp = np.where(vm.cpu().numpy())[0] if vm is not None else tp[:1000]
    va = torch.tensor(np.asarray(vp)[:1000], dtype=torch.long, device=dev)
    for k in args.shots:
        tr = torch.tensor(kshot_idx(y, tm, k, args.seed), dtype=torch.long, device=dev)
        for kind in args.gnns:
            acc = run(x, ei, y, tr, va, te, c, kind, dev)
            print(f"GNNBASE {args.dataset} {kind} {k} {acc:.2f}", flush=True)


if __name__ == "__main__":
    main()
