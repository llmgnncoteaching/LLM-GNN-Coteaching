#!/usr/bin/env python
"""GAugLLM-llama step 1: generate 4 augmented text views per node with a local
Llama-3-8B (vLLM OpenAI-compatible endpoint), SBERT-encode them, and save mixed
node features for the downstream GCL baseline.

Views (faithful to GAugLLM's descriptors):
  ORI  original raw_text (no generation)
  IDR  classification explanation from the node's own text
  SAS  summary of the node text informed by neighbor node texts
  SAR  classification explanation informed by neighbors
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _first_sentence(text, max_words=30):
    """Cheap 'title' proxy: first sentence / clause, truncated to max_words."""
    if not text:
        return ""
    text = str(text).strip().replace("\n", " ")
    for sep in (". ", "? ", "! ", "\t"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    words = text.split()
    return " ".join(words[:max_words])


def build_csr(edge_index, n):
    """Symmetric CSR adjacency for deterministic neighbor lookup."""
    from scipy.sparse import csr_matrix

    ei = edge_index.cpu().numpy()
    row = np.concatenate([ei[0], ei[1]])
    col = np.concatenate([ei[1], ei[0]])
    data = np.ones(row.shape[0], dtype=np.int8)
    A = csr_matrix((data, (row, col)), shape=(n, n))
    A.sum_duplicates()
    return A


def neighbor_context(node, A, raw_texts, k):
    """Titles of up to k 1-hop neighbors (deterministic: lowest node id first)."""
    nbrs = A.indices[A.indptr[node]:A.indptr[node + 1]]
    nbrs = sorted(int(x) for x in nbrs if int(x) != node)[:k]
    if not nbrs:
        return "(no neighbors)"
    return "\n".join(f"- {_first_sentence(raw_texts[j])}" for j in nbrs)


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
SYS = "You are a helpful assistant that explains text classification."


def make_messages(view, raw_text, labels, nctx):
    raw_text = str(raw_text)[:3000]
    labels = ", ".join(labels)
    if view == "IDR":
        user = (f"Text: {raw_text}\n\nClasses: {labels}.\n"
                "Explain in 2-3 sentences which class this text belongs to and "
                "why, based only on the text.")
    elif view == "SAS":
        user = (f"Text: {raw_text}\n\nTitles of neighboring nodes:\n{nctx}\n\n"
                "Write a 2-3 sentence summary of the text, informed by its "
                "neighbors.")
    elif view == "SAR":
        user = (f"Text: {raw_text}\n\nTitles of neighboring nodes:\n{nctx}\n\n"
                f"Classes: {labels}.\n"
                "Explain in 2-3 sentences which class this text belongs to, "
                "using the text and its neighbors.")
    else:
        raise ValueError(view)
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": user}]


def gen_one(client, model, view, raw_text, labels, nctx, max_tokens, raw_fallback):
    """One chat completion with retry/backoff; fall back to raw_text on failure."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=make_messages(view, raw_text, labels, nctx),
                max_tokens=max_tokens,
                temperature=0.2,
            )
            out = (resp.choices[0].message.content or "").strip()
            return out if out else raw_fallback
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return raw_fallback


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pt", required=True, help="path to our PyG-style .pt")
    ap.add_argument("--out", required=True, help="path to write x_aug.pt")
    ap.add_argument("--cache", default="", help="jsonl cache (default: <out>.jsonl)")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    ap.add_argument("--max_gen_tokens", type=int, default=256)
    ap.add_argument("--neighbors", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N nodes only")
    args = ap.parse_args()

    cache_path = args.cache or (args.out + ".jsonl")
    g = torch.load(args.pt, weights_only=False)
    raw_texts = list(g.raw_texts)
    n = len(raw_texts)
    if args.limit:
        n = min(n, args.limit)
    y = g.y.view(-1).long().cpu()
    labels = [str(x) for x in g.label_name]
    A = build_csr(g.edge_index, len(raw_texts))
    print(f"[gen] {args.dataset}: {n} nodes, {len(labels)} classes, "
          f"endpoint={args.endpoint} model={args.model}")

    # ---- resume cache -------------------------------------------------------
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[int(r["node_id"])] = r
                except Exception:
                    pass
    print(f"[gen] cache has {len(done)} nodes")

    todo = [i for i in range(n) if i not in done]
    from openai import OpenAI
    client = OpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=120)

    views = ["IDR", "SAS", "SAR"]

    def work(i):
        raw = raw_texts[i]
        nctx = neighbor_context(i, A, raw_texts, args.neighbors)
        out = {"node_id": i}
        for v in views:
            out[v.lower()] = gen_one(client, args.model, v, raw, labels, nctx,
                                     args.max_gen_tokens, _first_sentence(raw, 60))
        return out

    if todo:
        try:
            from tqdm import tqdm
        except Exception:
            def tqdm(x, **k):
                return x
        cf = open(cache_path, "a")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, i): i for i in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="generate"):
                r = fut.result()
                done[r["node_id"]] = r
                cf.write(json.dumps(r, ensure_ascii=False) + "\n")
                cf.flush()
        cf.close()

    n_generated = len(todo)
    n_cached = n - n_generated

    # ---- assemble the 4 view text lists ------------------------------------
    ori = [str(raw_texts[i]) for i in range(n)]
    idr = [done[i]["idr"] for i in range(n)]
    sas = [done[i]["sas"] for i in range(n)]
    sar = [done[i]["sar"] for i in range(n)]

    # ---- SBERT encode -------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sb = SentenceTransformer(args.sbert, device=dev)

    def enc(texts):
        return sb.encode(texts, batch_size=256, convert_to_tensor=True,
                         show_progress_bar=True).cpu().float()

    e_ori, e_idr, e_sas, e_sar = enc(ori), enc(idr), enc(sas), enc(sar)
    stack = torch.stack([e_ori, e_idr, e_sas, e_sar], dim=0)   # (4, N, 384)
    x_mean = stack.mean(dim=0)                                  # (N, 384)
    x_concat = torch.cat([e_ori, e_idr, e_sas, e_sar], dim=1)   # (N, 1536)

    def _mask(name):
        m = getattr(g, name, None)
        if torch.is_tensor(m):
            return m[:n].cpu().bool()
        return torch.zeros(n, dtype=torch.bool)

    torch.save({
        "x_mean": x_mean, "x_concat": x_concat, "x_ori": e_ori,
        "y": y[:n], "edge_index": g.edge_index.cpu().long(),
        "train_mask": _mask("train_mask"), "val_mask": _mask("val_mask"),
        "test_mask": _mask("test_mask"), "label_name": labels,
        "n_nodes": n,
    }, args.out)
    print(f"[gen] SAVED {args.out}: x_mean{tuple(x_mean.shape)} "
          f"x_concat{tuple(x_concat.shape)} | generated={n_generated} "
          f"cached={n_cached} test={int(_mask('test_mask').sum())}")


if __name__ == "__main__":
    main()
