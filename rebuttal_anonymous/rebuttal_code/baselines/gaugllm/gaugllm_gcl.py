#!/usr/bin/env python
"""GAugLLM-llama step 2: Graph Barlow Twins (GBT) contrastive pretrain on the
Llama-augmented SBERT features, then a k-shot linear probe.

Self-contained: torch + torch_geometric + sklearn + numpy only (no dgl / pecos).
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.b1 = nn.BatchNorm1d(hid)
        self.a1 = nn.PReLU()
        self.c2 = GCNConv(hid, out)
        self.b2 = nn.BatchNorm1d(out)
        self.a2 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self.a1(self.b1(self.c1(x, edge_index)))
        x = self.a2(self.b2(self.c2(x, edge_index)))
        return x


def bt_loss(z1, z2, lam=5e-3):
    """Graph Barlow Twins cross-correlation loss."""
    n = z1.shape[0]
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / n                       # (D, D)
    on = torch.diagonal(c).add_(-1).pow_(2).sum()
    off = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on + lam * off


def augment(x, edge_index, p_f=0.4, p_e=0.4):
    mask = (torch.rand(x.shape[1], device=x.device) >= p_f).float()
    x2 = x * mask
    ei2, _ = dropout_edge(edge_index, p=p_e)
    return x2, ei2


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def kshot_probe(emb, y, train_pool, test_idx, shots, seeds, n_classes):
    from sklearn.linear_model import LogisticRegression
    emb = emb.cpu().numpy()
    y = y.cpu().numpy()
    accs = []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        tr = []
        for c in range(n_classes):
            pool_c = train_pool[y[train_pool] == c]
            if len(pool_c) == 0:
                continue
            take = min(shots, len(pool_c))
            tr.extend(rng.choice(pool_c, size=take, replace=False).tolist())
        tr = np.array(tr)
        if len(tr) == 0:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(emb[tr], y[tr])
        pred = clf.predict(emb[test_idx])
        accs.append(float((pred == y[test_idx]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="x_aug.pt from step 1")
    ap.add_argument("--which", default="concat", choices=["mean", "concat", "ori"])
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test_cap", type=int, default=1000)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    d = torch.load(args.feat, weights_only=False)
    key = {"mean": "x_mean", "concat": "x_concat", "ori": "x_ori"}[args.which]
    x = d[key].float()
    y = d["y"].long()
    n = x.shape[0]
    edge_index = d["edge_index"].long()
    # keep only edges within [0, n) (safe if step-1 used --limit)
    m = (edge_index[0] < n) & (edge_index[1] < n)
    edge_index = edge_index[:, m]
    n_classes = int(y.max().item()) + 1

    # test set (cap), train pool = train_mask (fallback: all non-test/val)
    test_mask = d["test_mask"].bool()
    test_idx = torch.where(test_mask)[0].numpy()
    if args.test_cap and len(test_idx) > args.test_cap:
        test_idx = np.sort(np.random.RandomState(0).choice(
            test_idx, args.test_cap, replace=False))
    train_mask = d.get("train_mask")
    if train_mask is not None and train_mask.bool().any():
        train_pool = torch.where(train_mask.bool())[0].numpy()
    else:
        excl = set(test_idx.tolist())
        if d.get("val_mask") is not None:
            excl |= set(torch.where(d["val_mask"].bool())[0].numpy().tolist())
        train_pool = np.array([i for i in range(n) if i not in excl])

    x, edge_index = x.to(dev), edge_index.to(dev)
    enc = GCNEncoder(x.shape[1], args.emb_dim, args.emb_dim).to(dev)

    if not args.eval_only:
        opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.wd)
        enc.train()
        for ep in range(args.epochs):
            opt.zero_grad()
            x1, e1 = augment(x, edge_index)
            x2, e2 = augment(x, edge_index)
            z1, z2 = enc(x1, e1), enc(x2, e2)
            loss = bt_loss(z1, z2)
            loss.backward()
            opt.step()
            if ep % 100 == 0 or ep == args.epochs - 1:
                print(f"[gcl] epoch {ep} loss {loss.item():.4f}", flush=True)

    enc.eval()
    with torch.no_grad():
        emb = enc(x, edge_index)

    mean, std = kshot_probe(emb, y, train_pool, test_idx,
                            args.shots, args.seeds, n_classes)
    print(f"[eval] which={args.which} shots={args.shots} seeds={args.seeds} "
          f"|train_pool|={len(train_pool)} |test|={len(test_idx)}")
    print(f"GAUGLLM_LLAMA_ACC {args.which} {mean*100:.2f} {std*100:.2f}")


if __name__ == "__main__":
    main()
