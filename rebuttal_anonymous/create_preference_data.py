"""
Construct cross-round preference data for DPO/ORPO training.

Intuition: consecutive round pairs share the same batch of nodes.
- Odd round (R): model predicts BEFORE learning from this batch
- Even round (R+1): model predicts AFTER one round of co-teaching

For nodes where:
  - Round R: LLM and GNN DISAGREED (LLM was uncertain/wrong)
  - Round R+1: LLM and GNN AGREED (LLM corrected itself)
  
We construct:
  chosen  = Round R+1 LLM response (confirmed by GNN)
  rejected = Round R LLM response (was wrong, not confirmed)

This teaches the LLM: "after learning from GNN, your corrected prediction is better."
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from node_selection import load_llm_predictions_for_selected
from common import create_few_shot_dataset, set_seed


DIRECT_PROMPTS = {
    "arxiv": "Classify the above paper into one of the following categories: " + \
             "arxiv cs ai, arxiv cs cl, arxiv cs cc, arxiv cs ce, arxiv cs cg, " + \
             "arxiv cs cr, arxiv cs cv, arxiv cs cy, arxiv cs db, arxiv cs dc, " + \
             "arxiv cs dl, arxiv cs dm, arxiv cs ds, arxiv cs et, arxiv cs fl, " + \
             "arxiv cs gl, arxiv cs gr, arxiv cs gt, arxiv cs hc, arxiv cs ir, " + \
             "arxiv cs it, arxiv cs lg, arxiv cs lo, arxiv cs ma, arxiv cs mm, " + \
             "arxiv cs ms, arxiv cs na, arxiv cs ne, arxiv cs ni, arxiv cs oh, " + \
             "arxiv cs os, arxiv cs pf, arxiv cs pl, arxiv cs ro, arxiv cs sc, " + \
             "arxiv cs sd, arxiv cs se, arxiv cs si, arxiv cs sy, arxiv cs ar. " + \
             "Only reply the category.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Construct cross-round preference data for DPO"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--odd_round_dir", type=str, required=True,
                        help="Directory of the odd round (R)")
    parser.add_argument("--even_round_dir", type=str, required=True,
                        help="Directory of the even round (R+1)")
    parser.add_argument("--odd_round_predictions", type=str, required=True,
                        help="LLM predictions file from odd round")
    parser.add_argument("--even_round_predictions", type=str, required=True,
                        help="LLM predictions file from even round")
    parser.add_argument("--odd_round_gnn_model", type=str, required=True,
                        help="GNN model checkpoint from odd round")
    parser.add_argument("--even_round_gnn_model", type=str, required=True,
                        help="GNN model checkpoint from even round")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output preference data JSON")
    parser.add_argument("--odd_round", type=int, required=True)
    parser.add_argument("--even_round", type=int, required=True)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--path_prefix", type=str, default=".")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load graph data
    graph_data = create_few_shot_dataset(
        args.dataset, shots=args.shots, seed=args.seed,
        device=args.device, path_prefix=args.path_prefix,
    )
    graph_data = graph_data.to(args.device)
    num_classes = graph_data.y.max().item() + 1

    # Load shared node IDs (both rounds use same batch)
    odd_nodes_file = os.path.join(args.odd_round_dir, "sampled_nodes_ordered.json")
    even_nodes_file = os.path.join(args.even_round_dir, "sampled_nodes_ordered.json")

    with open(odd_nodes_file) as f:
        odd_node_ids = json.load(f)["selected_node_ids"]
    with open(even_nodes_file) as f:
        even_node_ids = json.load(f)["selected_node_ids"]

    # Verify same batch
    assert odd_node_ids == even_node_ids, \
        f"Node IDs mismatch! Odd has {len(odd_node_ids)}, even has {len(even_node_ids)}. " \
        f"First 3: {odd_node_ids[:3]} vs {even_node_ids[:3]}"
    node_ids = odd_node_ids
    print(f"Shared batch: {len(node_ids)} nodes")

    # Load LLM predictions for both rounds
    odd_llm_preds = load_llm_predictions_for_selected(
        args.odd_round_predictions, node_ids, graph_data
    )
    even_llm_preds = load_llm_predictions_for_selected(
        args.even_round_predictions, node_ids, graph_data
    )
    print(f"Odd round LLM predictions: {len(odd_llm_preds)}")
    print(f"Even round LLM predictions: {len(even_llm_preds)}")

    # Load GNN predictions for both rounds
    import torch
    from common import GNNEncoder

    def load_gnn_preds(model_path):
        model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
        ).to(args.device)
        model.load_state_dict(
            torch.load(model_path, map_location=args.device, weights_only=False)
        )
        model.eval()
        with torch.no_grad():
            logits = model(graph_data.x, graph_data.edge_index)
        return logits.argmax(dim=1)

    odd_gnn_preds = load_gnn_preds(args.odd_round_gnn_model)
    even_gnn_preds = load_gnn_preds(args.even_round_gnn_model)

    # Construct preference pairs
    prompt_template = DIRECT_PROMPTS.get(args.dataset.lower(), "")
    preference_data = []
    n_both_have = 0
    n_odd_disagree = 0
    n_even_agree = 0
    n_llm_changed = 0
    n_chosen_correct = 0
    n_rejected_correct = 0

    for nid in node_ids:
        nid = int(nid)

        # Both rounds must have valid LLM predictions
        if nid not in odd_llm_preds or nid not in even_llm_preds:
            continue
        n_both_have += 1

        odd_llm = odd_llm_preds[nid]
        even_llm = even_llm_preds[nid]
        odd_gnn = odd_gnn_preds[nid].item()
        even_gnn = even_gnn_preds[nid].item()

        # Condition 1: Odd round disagreed (LLM != GNN)
        if odd_llm == odd_gnn:
            continue
        n_odd_disagree += 1

        # Condition 2: Even round agreed (LLM == GNN)
        if even_llm != even_gnn:
            continue
        n_even_agree += 1

        # Condition 3: LLM changed its prediction
        if odd_llm == even_llm:
            continue
        n_llm_changed += 1

        # Construct preference pair
        text = graph_data.raw_texts[nid]
        chosen_label = graph_data.label_name[even_llm]
        rejected_label = graph_data.label_name[odd_llm]

        preference_data.append({
            "instruction": f"{text}\n{prompt_template}",
            "input": "",
            "chosen": chosen_label,
            "rejected": rejected_label,
            "node_id": nid,
            "odd_round": args.odd_round,
            "even_round": args.even_round,
        })

        # Track accuracy against ground truth
        gt = graph_data.y[nid].item()
        if even_llm == gt:
            n_chosen_correct += 1
        if odd_llm == gt:
            n_rejected_correct += 1

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)

    n_pairs = len(preference_data)
    print(f"\n{'='*60}")
    print(f"  Preference data: Round {args.odd_round} → {args.even_round}")
    print(f"  Shared batch: {len(node_ids)} nodes")
    print(f"  Both have LLM preds: {n_both_have}")
    print(f"  Odd round disagreed: {n_odd_disagree}")
    print(f"  Even round agreed: {n_even_agree}")
    print(f"  LLM changed prediction: {n_llm_changed}")
    print(f"  Final preference pairs: {n_pairs}")
    if n_pairs > 0:
        print(f"  Chosen (even agreed) accuracy: {n_chosen_correct}/{n_pairs} = {n_chosen_correct/n_pairs:.2%}")
        print(f"  Rejected (odd LLM) accuracy: {n_rejected_correct}/{n_pairs} = {n_rejected_correct/n_pairs:.2%}")
    print(f"  Saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
