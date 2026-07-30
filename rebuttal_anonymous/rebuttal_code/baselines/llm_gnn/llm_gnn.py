"""LLM-GNN baseline (Chen et al. 2023, label-free node classification):
LLM zero-shot annotates nodes (neighbor-aware prompt), a GCN trains on the LLM
labels (non-test pool), evaluated on the standard test set.
Annotation is zero-shot => one number per dataset (shot-independent), like Zero-shot/CoT.
Prints: LLMGNN <dataset> <acc> (llm_annot_acc <a>)."""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def build_csr(edge_index, n):
    import scipy.sparse as sp
    from torch_geometric.utils import to_scipy_sparse_matrix
    return to_scipy_sparse_matrix(edge_index, num_nodes=n).tocsr()


def llm_annotate(texts, label_names, adj, model_path, gpu_frac=0.6, k_nbr=0, direct_prompt=None):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_frac,
              max_model_len=4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    labs = ", ".join(label_names); low = [l.lower() for l in label_names]
    def nbr(i):
        js = adj.indices[adj.indptr[i]:adj.indptr[i + 1]][:k_nbr]
        return [str(texts[j])[:200] for j in js]
    def build(i):
        if k_nbr > 0:
            ctx = "\n".join(f"- {t}" for t in nbr(i))
            u = (f"Classes: {labs}.\nText: {str(texts[i])[:1800]}\n"
                 f"Neighboring node texts:\n{ctx}\n"
                 f"Using the text and its neighbors, reply with exactly one class name from the list.\nClass:")
        elif direct_prompt:
            u = f"{str(texts[i])[:2000]}\n\n{direct_prompt}"
        else:
            u = (f"Classes: {labs}.\nText: {str(texts[i])[:2000]}\n"
                 f"Reply with exactly one class name from the list.\nClass:")
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    prompts = [build(i) for i in range(len(texts))]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16))
    res = []
    for o in outs:
        a = o.outputs[0].text.strip().lower(); lab = 0
        for j, ln in enumerate(low):
            if ln in a: lab = j; break
        res.append(lab)
    return np.array(res, dtype=np.int64)


class GCN(nn.Module):
    def __init__(self, fi, h, c):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1, self.c2 = GCNConv(fi, h), GCNConv(h, c)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(x, ei)), 0.5, self.training)
        return self.c2(x, ei)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--model_path", default="/project/anon/rebuttal/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_cap", type=int, default=1000)
    ap.add_argument("--k_nbr", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.load(args.pt, weights_only=False)
    x = g.x.float(); ei = g.edge_index.long(); y = g.y.view(-1).long()
    n, c = x.size(0), int(y.max()) + 1
    labels = [str(l) for l in getattr(g, "label_name", [str(i) for i in range(c)])]
    adj = build_csr(ei, n)
    dp = None
    try:
        import sys; sys.path.insert(0, "/project/anon/rebuttal/port/LG-CoTeaching")
        from common.prompt import DIRECT_PROMPTS
        dp = DIRECT_PROMPTS.get(args.dataset)
    except Exception as e:
        print("no DIRECT_PROMPTS", e)
    pl = llm_annotate(list(g.raw_texts), labels, adj, args.model_path, k_nbr=args.k_nbr, direct_prompt=dp)
    annot_acc = (pl == y.cpu().numpy()).mean() * 100
    def mask(nm):
        m = getattr(g, nm, None); return m.cpu().bool() if torch.is_tensor(m) else None
    em, vm = mask("test_mask"), mask("val_mask")
    rng = np.random.RandomState(args.seed)
    tp = np.where(em.cpu().numpy())[0] if em is not None else np.arange(n)
    testidx = np.sort(rng.permutation(tp)[:args.test_cap])
    vp = np.where(vm.cpu().numpy())[0] if vm is not None else tp[:500]
    excl = set(testidx.tolist()) | set(np.asarray(vp).tolist())
    train_pool = np.array([i for i in range(n) if i not in excl], dtype=np.int64)
    xd, eid = x.to(dev), ei.to(dev)
    yl = torch.tensor(pl, device=dev)            # LLM labels (train signal)
    yg = torch.tensor(y.cpu().numpy(), device=dev)  # gold (eval only)
    tr = torch.tensor(train_pool, device=dev); va = torch.tensor(np.asarray(vp)[:500], dtype=torch.long, device=dev)
    te = torch.tensor(testidx, dtype=torch.long, device=dev)
    m = GCN(x.size(1), 128, c).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    best_va, best_te, pat = -1, 0, 0
    for ep in range(300):
        m.train(); opt.zero_grad()
        out = m(xd, eid); F.cross_entropy(out[tr], yl[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(xd, eid).argmax(1)
            va_a = (p[va] == yg[va]).float().mean().item()
            te_a = (p[te] == yg[te]).float().mean().item()
        if va_a >= best_va: best_va, best_te, pat = va_a, te_a, 0
        else:
            pat += 1
            if pat > 50: break
    print(f"LLMGNN {args.dataset} {best_te*100:.2f} (llm_annot_acc {annot_acc:.2f})", flush=True)


if __name__ == "__main__":
    main()
