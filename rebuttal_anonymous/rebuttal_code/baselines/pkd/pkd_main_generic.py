#!/usr/bin/env python
"""PKD baseline step 2: faithful, portable preference-based KD runner.

Reproduces arxiv_1.py's algorithm without its arxiv-hardwiring / missing-module
imports. Reads inputs written by pkd_prep.py from --datadir:
  {ds}_feature.npy, {ds}_edge_index.npy, {ds}_labels.npy,
  predicted_{ds}_labels_all.npy, {ds}_pca.npy, teacher_logits.npy (4,N,C),
  {ds}_trueidx.npy, {ds}_testidx.npy

Faithful to PKD:
  * node_selecting: pairwise KL among the 4 teachers, keep top --ratio% by total KL,
    unioned with the k-shot ground-truth train nodes (which use gold labels).
  * PPO agent (Model_P policy / Model_V value): per node, state = PCA embedding,
    action in {0,1,2,3} picks one teacher; assignment = one_hot(action,4).
  * distill(): selected_logit = assignment @ teacher_logits[:,i]; train the student
    GCN for --distill_epochs with KL(teacher_prob, student_prob) + NLL(student_prob,
    pseudo_label). reward = accuracy over the selected node set. PPO update per epoch.
  * final student GCN evaluated on the held-out test nodes -> PKD_LLAMA_ACC.

Only the student GCN class is imported from the PKD repo (models/gcn.py); everything
else is reimplemented here to stay portable. Model_P/Model_V match the repo
architecture (fc: in->1024->256->{4,1}) with the input dim set to the PCA dim.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d, 1024, bias=False), nn.Linear(1024, 256, bias=False), nn.Linear(256, 4, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d, 1024, bias=False), nn.Linear(1024, 256, bias=False), nn.Linear(256, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)


def kl_divergence(p, q):
    p, q = p + 1e-7, q + 1e-7
    return torch.sum(p * torch.log(p / q), dim=-1).mean()


def node_selecting(teacher, ratio):
    """pairwise-KL disagreement selection over teacher logits (4,N,C)."""
    def sm(a):
        e = np.exp(a - a.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    p = [sm(teacher[i]) + 1e-8 for i in range(4)]
    tot = np.zeros(teacher.shape[1])
    for i in range(4):
        for j in range(i + 1, 4):
            tot += np.sum(p[i] * np.log(p[i] / p[j]), axis=1)
    k = int(0.01 * ratio * teacher.shape[1])
    return np.argsort(tot)[::-1][:k].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--pkd_dir", required=True, help="PKD-main dir (for models/gcn.py)")
    ap.add_argument("--ratio", type=float, default=20.0, help="%% of nodes by teacher-disagreement")
    ap.add_argument("--outer_epochs", type=int, default=3)
    ap.add_argument("--distill_epochs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr_s", type=float, default=1e-3)
    ap.add_argument("--lr_rl", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max_nodes", type=int, default=8000, help="cap selected nodes for runtime")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, os.path.abspath(args.pkd_dir))
    from models.gcn import GCN                      # student architecture from the repo

    ds, dd = args.dataset, args.datadir
    x = torch.tensor(np.load(f"{dd}/{ds}_feature.npy"), dtype=torch.float32, device=dev)
    ei = torch.tensor(np.load(f"{dd}/{ds}_edge_index.npy"), dtype=torch.long, device=dev)
    y_true = torch.tensor(np.load(f"{dd}/{ds}_labels.npy"), dtype=torch.long, device=dev)
    pseudo = torch.tensor(np.load(f"{dd}/predicted_{ds}_labels_all.npy"), dtype=torch.long, device=dev)
    emb = torch.tensor(np.load(f"{dd}/{ds}_pca.npy"), dtype=torch.float32, device=dev)
    teacher_np = np.load(f"{dd}/teacher_logits.npy")            # (4,N,C)
    teacher = torch.tensor(teacher_np, dtype=torch.float32, device=dev)
    trueidx = np.load(f"{dd}/{ds}_trueidx.npy").tolist()
    testidx = torch.tensor(np.load(f"{dd}/{ds}_testidx.npy"), dtype=torch.long, device=dev)
    n, fi, c = x.size(0), x.size(1), teacher.size(2)
    print(f"[pkd] {ds}: N={n} F={fi} C={c} pca={emb.size(1)} ratio={args.ratio}%")

    # ---- node selection: teacher-disagreement top-ratio ∪ k-shot gold ------ #
    sel = node_selecting(teacher_np, args.ratio)
    index_list = sorted(set(trueidx) | set(sel))
    labels = pseudo.clone()
    for i in trueidx:                                          # gold overrides on labeled
        labels[i] = y_true[i]
    if len(index_list) > args.max_nodes:                       # runtime cap (log it)
        rng = np.random.RandomState(args.seed)
        keep = set(trueidx) | set(rng.choice(sel, args.max_nodes - len(trueidx), replace=False).tolist())
        index_list = sorted(keep)
        print(f"[pkd] capped selected nodes to {len(index_list)} (of {len(set(trueidx)|set(sel))})")
    print(f"[pkd] distillation node set = {len(index_list)} (gold {len(trueidx)} + disagreement)")

    student = GCN(fi, args.hidden, c, 2, 0.5).to(dev)
    policy, value = PolicyNet(emb.size(1)).to(dev), ValueNet(emb.size(1)).to(dev)
    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr_s, weight_decay=1e-5)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_rl, weight_decay=1e-4)
    opt_v = torch.optim.Adam(value.parameters(), lr=args.lr_rl, weight_decay=1e-4)

    def student_out():
        return student(x, ei)                                 # GCNConv accepts edge_index

    def test_acc():
        student.eval()
        with torch.no_grad():
            _, pred, _ = student_out()
        return (pred[testidx] == y_true[testidx]).float().mean().item() * 100

    def distill(i, assignment):
        sel_logit = assignment @ teacher[:, i, :]             # (C,)
        tprob = F.softmax(sel_logit, dim=-1)
        for _ in range(args.distill_epochs):
            student.train(); opt_s.zero_grad()
            slog, _, _ = student_out()
            sprob = F.softmax(slog[i].float(), dim=-1)
            ce = F.nll_loss(sprob.unsqueeze(0), labels[i].unsqueeze(0))   # faithful PKD quirk
            dl = kl_divergence(tprob, sprob)
            (dl + ce).backward(); opt_s.step()
        student.eval()
        with torch.no_grad():
            _, pred, _ = student_out()
        idx = torch.tensor(index_list, device=dev)
        return (pred[idx] == labels[idx]).float().mean().item() * 100

    print(f"[pkd] init student test acc {test_acc():.2f}")
    for ep in range(args.outer_epochs):
        traj = []
        for i in index_list:
            st = emb[i]
            probs = policy(st); dist = Categorical(probs)
            action = dist.sample(); logp = dist.log_prob(action)
            val = value(st)
            assign = F.one_hot(action, num_classes=4).float().to(dev)
            reward = distill(i, assign)
            traj.append((st.detach(), action.detach(), logp.detach(), float(reward), val.detach()))
        # ---- PPO update (faithful: clipped surrogate + value MSE) ---------- #
        rets, cum = [], 0.0
        for (_, _, _, r, _) in reversed(traj):
            cum = r + args.gamma * cum; rets.insert(0, cum)
        rets = torch.tensor(rets, dtype=torch.float32, device=dev)
        rets = (rets - rets.mean()) / (rets.std() + 1e-6)
        for (st, action, old_logp, _, val), R in zip(traj, rets):
            adv = (R - val.squeeze()).detach()
            probs = policy(st); dist = Categorical(probs)
            new_logp = dist.log_prob(action)
            ratio_ = torch.exp(new_logp - old_logp)
            s1, s2 = ratio_ * adv, torch.clamp(ratio_, 1 - args.clip, 1 + args.clip) * adv
            ploss = -torch.min(s1, s2)
            opt_p.zero_grad(); ploss.backward(); opt_p.step()
            vloss = F.mse_loss(value(st).squeeze(), R)
            opt_v.zero_grad(); vloss.backward(); opt_v.step()
        print(f"[pkd] outer epoch {ep+1}/{args.outer_epochs}  student test acc {test_acc():.2f}")

    acc = test_acc()
    print(f"PKD_LLAMA_ACC {ds} {acc:.2f}")


if __name__ == "__main__":
    main()
