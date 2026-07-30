"""
Create Cross-Filter data for bidirectional pseudo-label exchange.

Implements the core Co-Teaching principle (Han et al., NeurIPS 2018) adapted
for heterogeneous GNN-LLM pairs on text-attributed graphs:

  Co-Teaching insight: Each network selects "clean" samples for the other.
  Two networks with different inductive biases make different mistakes,
  so cross-selection filters noise that self-selection cannot.

  Our adaptation:
    - GNN (structural bias) selects clean pseudo-labels for LLM (semantic bias)
    - LLM (semantic bias) selects clean pseudo-labels for GNN (structural bias)
    - "Clean" = high confidence from the selecting model (small-loss ≈ high confidence)
    - Selection ratio R(t) increases over rounds as both models improve

Cross-selection pipeline:
  1. Both models predict on the same unlabeled nodes
  2. GNN sorts its predictions by confidence → top-R% become labels for LLM
  3. LLM's agreed predictions become labels for GNN (LLM confidence is binary: parseable or not)
  4. Agreed nodes (both predict same label) are trusted by both sides
  5. Neither model trains on its own selected samples — only the peer's selections

Usage:
    python create_co_teaching_data.py \
        --dataset arxiv --shots 3 --seed 42 \
        --pretrained_model results/co_teaching/gnn_round0.pt \
        --llm_predictions path/to/llm_preds.jsonl \
        --selected_nodes_path path/to/selected_nodes_ordered.json \
        --sft_output_path  path/to/gnn_selects_for_llm.json \
        --gnn_pseudo_label_path path/to/llm_selects_for_gnn.json \
        --select_ratio 0.3 --round 1
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from common import (
    GNNEncoder,
    DIRECT_PROMPTS,
    create_few_shot_dataset,
    load_graph_dataset,
    set_seed,
)
from node_selection import (
    find_agreed_and_disagreed_nodes,
    load_llm_predictions_for_selected,
    load_llm_logprobs_for_selected,
    retrain_gnn_on_agreed,
)


# -------------------------------------------------------------------------
# Core Co-Teaching: small-loss (= high-confidence) cross-selection
# -------------------------------------------------------------------------

def cross_select_gnn_for_llm(
    gnn_predictions, llm_predictions, selected_node_ids,
    graph_data, select_ratio,
):
    """
    GNN selects clean pseudo-labels FOR the LLM (cross-selection).

    True Co-Teaching (Bo Han): each model selects its small-loss samples.
    GNN treats its own prediction as the "label", computes CE loss,
    and selects nodes with smallest loss (= highest confidence) for LLM.

    Label = GNN's prediction (this is GNN's knowledge gift to LLM).
    """
    gnn_probs = F.softmax(gnn_predictions, dim=1)
    gnn_preds = gnn_predictions.argmax(dim=1)
    num_classes = gnn_predictions.shape[1]

    candidates = []
    for nid in selected_node_ids:
        nid = int(nid)
        gnn_pred = gnn_preds[nid].item()

        # Validate prediction is in range
        if not (0 <= gnn_pred < num_classes):
            continue

        gnn_conf = gnn_probs[nid, gnn_pred].item()

        # Skip near-zero confidence (degenerate softmax)
        if gnn_conf < 1e-6:
            continue

        llm_pred = llm_predictions.get(nid, -1)
        is_agreed = (gnn_pred == llm_pred)

        # Small-loss: CE loss = -log(softmax[predicted_class])
        gnn_loss = -torch.log(gnn_probs[nid, gnn_pred] + 1e-8).item()

        # Score: negative loss (so higher = smaller loss = more confident)
        score = -gnn_loss
        candidates.append((nid, gnn_pred, gnn_conf, is_agreed, score, gnn_loss))

    # Sort by score descending (= loss ascending = smallest loss first)
    candidates.sort(key=lambda x: x[4], reverse=True)

    n_select = max(1, int(len(candidates) * select_ratio))
    selected = candidates[:n_select]

    selected_labels = {}
    n_agreed = 0
    for nid, pred, conf, agreed, score, loss in selected:
        selected_labels[nid] = pred
        if agreed:
            n_agreed += 1

    correct = sum(1 for nid, lid in selected_labels.items()
                  if graph_data.y[nid].item() == lid)
    total = len(selected_labels)
    acc = correct / total if total > 0 else 0.0

    stats = {
        "total_candidates": len(candidates),
        "n_selected": total,
        "n_agreed_in_selected": n_agreed,
        "n_disagreed_in_selected": total - n_agreed,
        "select_ratio": select_ratio,
        "pseudo_label_accuracy": round(acc, 4),
        "avg_gnn_loss": round(sum(c[5] for c in selected) / max(len(selected), 1), 4),
        "avg_gnn_confidence": round(sum(c[2] for c in selected) / max(len(selected), 1), 4),
    }
    return selected_labels, stats


def cross_select_llm_for_gnn(
    gnn_predictions, llm_predictions, selected_node_ids,
    graph_data, select_ratio, llm_logprobs=None,
):
    """
    LLM selects clean pseudo-labels FOR the GNN (cross-selection).

    True Co-Teaching (Bo Han): LLM selects its small-loss samples.
    LLM loss ≈ -logprob of the discriminative token.
    Only considers nodes where LLM produced a valid, parseable label.

    Label = LLM's prediction (this is LLM's knowledge gift to GNN).
    """
    gnn_probs = F.softmax(gnn_predictions, dim=1)
    gnn_preds = gnn_predictions.argmax(dim=1)
    num_classes = gnn_predictions.shape[1]

    n_skipped_no_parse = 0
    n_skipped_no_logprob = 0
    n_skipped_bad_label = 0
    candidates = []

    for nid in selected_node_ids:
        nid = int(nid)

        # Skip if LLM couldn't parse a valid label
        if nid not in llm_predictions:
            n_skipped_no_parse += 1
            continue

        llm_pred = llm_predictions[nid]

        # Skip if predicted label is out of range
        if not (0 <= llm_pred < num_classes):
            n_skipped_bad_label += 1
            continue

        # Skip if no logprob available (can't estimate confidence)
        if not llm_logprobs or nid not in llm_logprobs:
            n_skipped_no_logprob += 1
            continue

        llm_conf = llm_logprobs[nid]

        # Skip extreme outliers (likely garbage generation)
        if llm_conf < -10.0 or llm_conf != llm_conf:  # nan check
            n_skipped_no_logprob += 1
            continue

        gnn_pred = gnn_preds[nid].item()
        gnn_conf = gnn_probs[nid, gnn_pred].item()
        is_agreed = (gnn_pred == llm_pred)

        # LLM's own small-loss score (higher logprob = smaller loss)
        score = llm_conf
        candidates.append((nid, llm_pred, gnn_conf, is_agreed, score, llm_conf))

    candidates.sort(key=lambda x: x[4], reverse=True)

    n_select = max(1, int(len(candidates) * select_ratio))
    selected = candidates[:n_select]

    selected_labels = {}
    n_agreed = 0
    for nid, pred, gnn_conf, agreed, score, llm_conf in selected:
        selected_labels[nid] = pred
        if agreed:
            n_agreed += 1

    correct = sum(1 for nid, lid in selected_labels.items()
                  if graph_data.y[nid].item() == lid)
    total = len(selected_labels)
    acc = correct / total if total > 0 else 0.0

    avg_llm_conf = sum(c[5] for c in selected) / max(len(selected), 1)

    stats = {
        "total_candidates": len(candidates),
        "n_selected": total,
        "n_agreed_in_selected": n_agreed,
        "n_disagreed_in_selected": total - n_agreed,
        "n_skipped_no_parse": n_skipped_no_parse,
        "n_skipped_no_logprob": n_skipped_no_logprob,
        "select_ratio": select_ratio,
        "pseudo_label_accuracy": round(acc, 4),
        "avg_llm_logprob": round(avg_llm_conf, 4),
    }
    return selected_labels, stats


# -------------------------------------------------------------------------
# Output: GNN → LLM SFT data
# -------------------------------------------------------------------------

def extract_title(text, max_len=120):
    """Extract title from ogbn-arxiv raw_text format.
    
    ogbn-arxiv format: 'title. abstract text #R##N# more abstract text'
    Strategy: take text before #R##N# separator, then cap at max_len.
    This gets the title + possibly start of abstract, which is fine for neighbor context.
    """
    if '#R##N#' in text:
        text = text.split('#R##N#')[0].strip()
    # Cap length
    if len(text) > max_len:
        # Try to break at a word boundary
        cut = text[:max_len].rfind(' ')
        if cut > max_len // 2:
            text = text[:cut] + "..."
        else:
            text = text[:max_len] + "..."
    return text.strip()


def build_adj_cache(graph_data):
    """Build CSR adjacency matrices for fast neighbor lookup."""
    from torch_geometric.utils import to_scipy_sparse_matrix
    adj = to_scipy_sparse_matrix(graph_data.edge_index, num_nodes=graph_data.num_nodes).tocsr()
    adj2 = (adj @ adj).tocsr()
    return adj, adj2


def get_neighbor_info(nid, raw_texts, adj, adj2, max_1hop=5, max_2hop=5):
    """Get 1-hop and 2-hop neighbor titles using precomputed CSR adjacency.
    
    Deterministic: takes first max_N neighbors from CSR order (sorted by node id).
    This ensures consistency between training and inference prompts.
    """
    hop1 = adj[nid].indices.tolist()
    if not hop1:
        return ""
    hop1_sampled = hop1[:max_1hop]

    hop1_set = set(hop1)
    hop2 = [n for n in adj2[nid].indices.tolist() if n != nid and n not in hop1_set]
    hop2_sampled = hop2[:max_2hop]

    lines = []
    t1 = [f"[{i+1}] {extract_title(raw_texts[idx])}" for i, idx in enumerate(hop1_sampled)]
    if t1:
        lines.append("1-hop neighbors: " + " ".join(t1))
    if hop2_sampled:
        t2 = [f"[{i+1}] {extract_title(raw_texts[idx])}" for i, idx in enumerate(hop2_sampled)]
        lines.append("2-hop neighbors: " + " ".join(t2))

    return "\n".join(lines)


def write_sft_data(
    selected_labels, graph_data, dataset_name, output_path,
    include_train_nodes=True, anchor_repeat=1, use_neighbor_info=False,
    adj=None, adj2=None,
):
    """Write SFT JSON from cross-selected labels (GNN selects for LLM).
    
    anchor_repeat: number of times to repeat the ground-truth anchor samples.
    """
    prompt = DIRECT_PROMPTS.get(dataset_name.lower(), "")
    if not prompt:
        raise ValueError(f"No prompt template for: {dataset_name}")

    dataset = []

    # Include original few-shot train nodes as anchors (repeated anchor_repeat times)
    if include_train_nodes:
        train_indices = torch.where(graph_data.train_mask)[0].cpu().tolist()
        for _ in range(anchor_repeat):
            for idx in train_indices:
                text = graph_data.raw_texts[idx]
                label = graph_data.label_name[graph_data.y[idx].cpu().item()]
                neighbor_ctx = get_neighbor_info(idx, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
                human_text = f"{text}\n{neighbor_ctx}\n{prompt}" if neighbor_ctx else f"{text}\n{prompt}"
                dataset.append({
                    "node_id": idx,
                    "is_anchor": True,
                    "conversations": [
                        {"from": "human", "value": human_text},
                        {"from": "gpt", "value": label},
                    ]
                })
    n_anchor = len(dataset)

    # Cross-selected pseudo-labels
    for nid, lid in selected_labels.items():
        if lid < 0 or lid >= len(graph_data.label_name):
            continue
        text = graph_data.raw_texts[nid]
        label = graph_data.label_name[lid]
        neighbor_ctx = get_neighbor_info(nid, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
        human_text = f"{text}\n{neighbor_ctx}\n{prompt}" if neighbor_ctx else f"{text}\n{prompt}"
        dataset.append({
            "node_id": int(nid),
            "is_anchor": False,
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": label},
            ]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    print(f"  GNN→LLM SFT data: {len(dataset)} samples ({n_anchor} anchor + {len(dataset)-n_anchor} pseudo) → {output_path}")
    return len(dataset)


# -------------------------------------------------------------------------
# Output: LLM → GNN pseudo-labels
# -------------------------------------------------------------------------

def write_gnn_pseudo_labels(
    selected_labels, graph_data, output_path,
):
    """Write pseudo-label JSON from cross-selected labels (LLM selects for GNN)."""
    pseudo = {str(nid): int(lid) for nid, lid in selected_labels.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"pseudo_labels": pseudo}, f, indent=2)

    correct = sum(1 for nid, lid in selected_labels.items()
                  if graph_data.y[nid].item() == lid)
    total = len(selected_labels)
    acc = correct / total if total > 0 else 0.0
    print(f"  LLM→GNN pseudo-labels: {total} nodes → {output_path}")
    print(f"  LLM→GNN pseudo-label accuracy: {correct}/{total} = {acc:.4f}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Filter: bidirectional pseudo-label cross-selection"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--selected_nodes_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--llm_predictions", type=str, required=True)
    parser.add_argument("--sft_output_path", type=str, required=True,
                        help="GNN selects for LLM: SFT data output")
    parser.add_argument("--gnn_pseudo_label_path", type=str, required=True,
                        help="LLM selects for GNN: pseudo-label output")
    parser.add_argument("--round", type=int, default=1,
                        help="Current co-teaching round (for logging)")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--gnn_type", type=str, default="GCN",
                        choices=["GCN", "GAT", "SAGE", "SGConv"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--path_prefix", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retrain_on_agreed", type=int, default=1,
                        help="Retrain GNN on agreed nodes before selection (sharpens confidence)")
    parser.add_argument("--anchor_repeat", type=int, default=1,
                        help="Repeat ground-truth anchor samples N times in SFT data")
    parser.add_argument("--use_cumulative", type=int, default=0,
                        help="Write agreed-only files for historical accumulation (0=off, 1=on)")
    parser.add_argument("--num_rounds", type=int, default=20,
                        help="Total number of co-teaching rounds (for R(t) schedule)")
    parser.add_argument("--rt_min", type=float, default=0.2,
                        help="R(t) minimum (starting selection ratio)")
    parser.add_argument("--rt_max", type=float, default=0.6,
                        help="R(t) maximum (ending selection ratio)")
    parser.add_argument("--use_neighbor_info", type=int, default=0,
                        help="Include neighbor text snippets in LLM prompt (0=off, 1=on)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load graph
    if args.shots:
        graph_data = create_few_shot_dataset(
            args.dataset, shots=args.shots, seed=args.seed,
            device=args.device, path_prefix=args.path_prefix
        )
    else:
        graph_data = load_graph_dataset(args.dataset, device=args.device,
                                        path_prefix=args.path_prefix)

    # Dataset-specific num_classes (products subset has 47 classes but some
    # few-shot train splits may not cover all classes, giving wrong y.max())
    if args.dataset == "ogbn-products_subset":
        num_classes = 47
    else:
        num_classes = graph_data.y.max().item() + 1

    # Load selected node IDs (this round's random sample)
    with open(args.selected_nodes_path, 'r') as f:
        selected_nodes_ids = json.load(f)["selected_node_ids"]

    # Load GNN
    gnn_model = GNNEncoder(
        input_dim=graph_data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        n_layers=args.n_layers,
        gnn_type=args.gnn_type,
    ).to(args.device)
    gnn_model.load_state_dict(
        torch.load(args.pretrained_model, map_location=args.device, weights_only=False)
    )
    gnn_model.eval()

    with torch.no_grad():
        gnn_predictions = gnn_model(graph_data.x, graph_data.edge_index)

    # Load LLM predictions
    print(f"Loading LLM predictions from {args.llm_predictions}")
    llm_predictions = load_llm_predictions_for_selected(
        args.llm_predictions, selected_nodes_ids, graph_data
    )
    print(f"LLM predictions mapped: {len(llm_predictions)}/{len(selected_nodes_ids)}")

    covered_ids = [int(nid) for nid in selected_nodes_ids if int(nid) in llm_predictions]

    # Load LLM logprobs for confidence-based selection
    llm_logprobs = load_llm_logprobs_for_selected(
        args.llm_predictions, selected_nodes_ids
    )
    print(f"LLM logprobs loaded: {len(llm_logprobs)}/{len(selected_nodes_ids)}")

    # Optional: retrain GNN on agreed nodes to sharpen confidence
    if args.retrain_on_agreed:
        agreed, _ = find_agreed_and_disagreed_nodes(
            gnn_predictions, llm_predictions, graph_data
        )
        agreed_filtered = {nid: agreed[nid] for nid in covered_ids if nid in agreed}
        if len(agreed_filtered) > 0:
            print(f"Sharpening GNN on {len(agreed_filtered)} agreed nodes...")
            retrain_gnn_on_agreed(gnn_model, graph_data, agreed_filtered,
                                  args.device, lr=1e-3, epochs=50)
            with torch.no_grad():
                gnn_predictions = gnn_model(graph_data.x, graph_data.edge_index)

    # === Compute R(t): linear schedule from R_min to R_max ===
    gnn_preds = gnn_predictions.argmax(dim=1)
    n_agreed = sum(1 for nid in covered_ids
                   if gnn_preds[nid].item() == llm_predictions.get(nid, -1))
    agreed_rate = n_agreed / max(len(covered_ids), 1)

    if args.num_rounds > 1:
        select_ratio = args.rt_min + (args.rt_max - args.rt_min) * (args.round - 1) / (args.num_rounds - 1)
    else:
        select_ratio = args.rt_min

    print(f"\n{'='*60}")
    print(f"  Cross-Filter Round {args.round}")
    print(f"  Mini-batch: {len(covered_ids)} nodes")
    print(f"  Agreed: {n_agreed}/{len(covered_ids)} = {agreed_rate:.2%}")
    print(f"  R(t) = {select_ratio:.2%} (linear: {args.rt_min} → {args.rt_max} over {args.num_rounds} rounds)")
    print(f"{'='*60}")

    # === Build adjacency cache for neighbor info (if needed) ===
    adj, adj2 = None, None
    if args.use_neighbor_info:
        print("Building adjacency cache for neighbor info...")
        adj, adj2 = build_adj_cache(graph_data)
        print(f"  adj: {adj.shape}, nnz={adj.nnz}; adj2 nnz={adj2.nnz}")

    # === Cross-selection 1: GNN selects clean labels FOR LLM ===
    print(f"\n--- GNN cross-selects for LLM (top {select_ratio:.0%}) ---")
    gnn_for_llm, gnn_stats = cross_select_gnn_for_llm(
        gnn_predictions, llm_predictions, covered_ids,
        graph_data, select_ratio,
    )
    print(f"  Selected: {gnn_stats['n_selected']}/{gnn_stats['total_candidates']} "
          f"(agreed: {gnn_stats['n_agreed_in_selected']}, "
          f"acc: {gnn_stats['pseudo_label_accuracy']:.4f}, "
          f"avg_conf: {gnn_stats['avg_gnn_confidence']:.4f})")

    write_sft_data(
        gnn_for_llm, graph_data, args.dataset, args.sft_output_path,
        include_train_nodes=True, anchor_repeat=args.anchor_repeat,
        use_neighbor_info=bool(args.use_neighbor_info), adj=adj, adj2=adj2,
    )

    # === Cross-selection 2: LLM selects clean labels FOR GNN ===
    print(f"\n--- LLM cross-selects for GNN (top {select_ratio:.0%}) ---")
    llm_for_gnn, llm_stats = cross_select_llm_for_gnn(
        gnn_predictions, llm_predictions, covered_ids,
        graph_data, select_ratio, llm_logprobs=llm_logprobs,
    )
    print(f"  Selected: {llm_stats['n_selected']}/{llm_stats['total_candidates']} "
          f"(agreed: {llm_stats['n_agreed_in_selected']}, "
          f"acc: {llm_stats['pseudo_label_accuracy']:.4f}, "
          f"avg_logprob: {llm_stats.get('avg_llm_logprob', 'N/A')})")

    write_gnn_pseudo_labels(llm_for_gnn, graph_data, args.gnn_pseudo_label_path)

    # === Write agreed-only files for cumulative accumulation ===
    if args.use_cumulative:
        gnn_preds_for_sel = gnn_predictions.argmax(dim=1)

        # GNN→LLM: agreed subset (with anchor)
        gnn_for_llm_agreed = {
            nid: lid for nid, lid in gnn_for_llm.items()
            if gnn_preds_for_sel[nid].item() == llm_predictions.get(nid, -1)
        }
        agreed_sft_path = args.sft_output_path.replace('.json', '_agreed.json')
        write_sft_data(
            gnn_for_llm_agreed, graph_data, args.dataset, agreed_sft_path,
            include_train_nodes=True, anchor_repeat=args.anchor_repeat,
            use_neighbor_info=bool(args.use_neighbor_info), adj=adj, adj2=adj2,
        )
        print(f"  Agreed SFT: {len(gnn_for_llm_agreed)}/{len(gnn_for_llm)} → {agreed_sft_path}")

        # LLM→GNN: agreed subset
        llm_for_gnn_agreed = {
            nid: lid for nid, lid in llm_for_gnn.items()
            if gnn_preds_for_sel[nid].item() == llm_predictions.get(nid, -1)
        }
        agreed_pseudo_path = args.gnn_pseudo_label_path.replace('.json', '_agreed.json')
        agreed_pseudo = {str(nid): int(lid) for nid, lid in llm_for_gnn_agreed.items()}
        os.makedirs(os.path.dirname(agreed_pseudo_path), exist_ok=True)
        with open(agreed_pseudo_path, 'w') as f:
            json.dump({"pseudo_labels": agreed_pseudo}, f)
        print(f"  Agreed pseudo: {len(llm_for_gnn_agreed)}/{len(llm_for_gnn)} → {agreed_pseudo_path}")

    # Save round stats
    stats_path = args.gnn_pseudo_label_path.replace('.json', '_stats.json')

    with open(stats_path, 'w') as f:
        json.dump({
            "round": args.round,
            "batch_size": len(covered_ids),
            "n_agreed": n_agreed,
            "agreed_rate": round(agreed_rate, 4),
            "select_ratio_Rt": round(select_ratio, 4),
            "gnn_selects_for_llm": gnn_stats,
            "llm_selects_for_gnn": llm_stats,
        }, f, indent=2)

    print(f"\nCross-Filter round {args.round} complete.")


if __name__ == "__main__":
    main()
