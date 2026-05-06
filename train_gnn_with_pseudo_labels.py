"""
Train GNN with LLM pseudo-labels (LLM → GNN direction in Co-Teaching).

Combines the original few-shot training loss with a pseudo-label loss:
    Loss = (1 - alpha) * CE(original_train_mask) + alpha * CE(pseudo_mask)

Usage:
    python train_gnn_with_pseudo_labels.py \
        --dataset cora --shots 5 --seed 42 \
        --pseudo_label_path path/to/llm_teaches_gnn.json \
        --pretrained_model results/GNN/cora_5_shot_best_model_run0.pt \
        --save_path results/co_teaching/round1_gnn.pt \
        --alpha 0.5
"""

import argparse
import json
import time
import os

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import (
    GNNEncoder,
    set_seed,
    compute_acc_and_f1,
    create_few_shot_dataset,
)


def gnn_train_co_teaching(model, data, optimizer, pseudo_mask, pseudo_y, alpha):
    """One training step with combined loss.
    
    Instance-wise: each set averaged separately, then alpha-weighted combine.
    L = (1-α) * mean_CE(anchor) + α * mean_CE(pseudo)
    """
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)

    # Original few-shot loss (sample-wise mean over anchor)
    loss_orig = F.cross_entropy(output[data.train_mask], data.y[data.train_mask],
                                reduction='mean')

    # Pseudo-label loss (sample-wise mean over pseudo)
    if pseudo_mask.any():
        loss_pseudo = F.cross_entropy(output[pseudo_mask], pseudo_y[pseudo_mask],
                                      reduction='mean')
    else:
        loss_pseudo = torch.tensor(0.0, device=output.device)

    # Alpha-weighted combine of the two MEANS (not sums)
    loss = (1 - alpha) * loss_orig + alpha * loss_pseudo
    loss.backward()
    optimizer.step()

    return float(loss), float(loss_orig), float(loss_pseudo)


@torch.no_grad()
def gnn_test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)

    accuracy, macro_f1_scores, micro_f1_scores = [], [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        if mask.sum() == 0:
            accuracy.append(0.0)
            macro_f1_scores.append(0.0)
            micro_f1_scores.append(0.0)
        else:
            acc, macro_f1, micro_f1 = compute_acc_and_f1(
                pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()
            )
            accuracy.append(acc)
            macro_f1_scores.append(macro_f1)
            micro_f1_scores.append(micro_f1)

    return accuracy, macro_f1_scores, micro_f1_scores


@torch.no_grad()
def evaluate_pseudo_label_acc(model, data, pseudo_mask, pseudo_y):
    """Check how well the GNN now agrees with the pseudo-labels."""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    if pseudo_mask.any():
        correct = (pred[pseudo_mask] == pseudo_y[pseudo_mask]).sum().item()
        total = pseudo_mask.sum().item()
        return correct / total if total > 0 else 0.0
    return 0.0


def load_pseudo_labels(pseudo_label_path, num_nodes, device):
    """Load pseudo-labels from JSON and build mask + label tensors."""
    with open(pseudo_label_path, 'r') as f:
        data = json.load(f)

    pseudo_dict = data["pseudo_labels"]
    pseudo_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    pseudo_y = torch.zeros(num_nodes, dtype=torch.long, device=device)

    for nid_str, lid in pseudo_dict.items():
        nid = int(nid_str)
        if 0 <= nid < num_nodes:
            pseudo_mask[nid] = True
            pseudo_y[nid] = int(lid)

    return pseudo_mask, pseudo_y


def print_dataset_stats(graph_data, pseudo_mask):
    total_nodes = graph_data.x.shape[0]
    num_train = int(torch.sum(graph_data.train_mask))
    num_val = int(torch.sum(graph_data.val_mask))
    num_test = int(torch.sum(graph_data.test_mask))
    num_pseudo = int(torch.sum(pseudo_mask))

    print(f"Data shape: features {graph_data.x.shape}, edges {graph_data.edge_index.shape}")
    print(f"Total nodes: {total_nodes}")
    print(f"Train nodes: {num_train} ({num_train/total_nodes:.2%})")
    print(f"Validation nodes: {num_val} ({num_val/total_nodes:.2%})")
    print(f"Test nodes: {num_test} ({num_test/total_nodes:.2%})")
    print(f"Pseudo-labeled nodes: {num_pseudo} ({num_pseudo/total_nodes:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN with LLM pseudo-labels (Co-Teaching)"
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--pseudo_label_path", type=str, required=True,
                        help="Path to LLM→GNN pseudo-label JSON")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pre-trained GNN checkpoint (continue training)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Where to save the updated GNN model")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--gnn_type", type=str, default="GCN",
                        choices=["GCN", "GAT", "SAGE", "SGConv"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of pseudo-label loss (0=only original, 1=only pseudo)")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate for teacher model (0=disabled)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--path_prefix", type=str, default=".")
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    set_seed(args.seed)

    # Load graph data with same split
    graph_data = create_few_shot_dataset(
        args.dataset, shots=args.shots, seed=args.seed,
        device=device, path_prefix=args.path_prefix,
    )
    graph_data = graph_data.to(device)
    num_classes = graph_data.y.max().item() + 1

    # Load pseudo-labels
    pseudo_mask, pseudo_y = load_pseudo_labels(
        args.pseudo_label_path, graph_data.num_nodes, device
    )
    print_dataset_stats(graph_data, pseudo_mask)

    # Build GNN
    gnn_model = GNNEncoder(
        input_dim=graph_data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        n_layers=args.n_layers,
        gnn_type=args.gnn_type,
        dropout=args.dropout,
    ).to(device)

    # Optionally load pre-trained weights
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pre-trained GNN from {args.pretrained_model}")
        gnn_model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
    print(f"[GNN] Number of parameters: {trainable_params:,}")
    print(f"[GNN] Alpha (pseudo weight): {args.alpha}")

    # EMA model (Mean Teacher) — persists across rounds
    import copy
    ema_model = None
    if args.ema_decay > 0:
        ema_save_path = args.save_path.replace('.pt', '_ema.pt')
        # Try to load previous round's EMA (for cross-round continuity)
        prev_ema_path = args.pretrained_model.replace('.pt', '_ema.pt') if args.pretrained_model else None
        if prev_ema_path and os.path.exists(prev_ema_path):
            ema_model = GNNEncoder(
                input_dim=graph_data.x.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                n_layers=args.n_layers,
                gnn_type=args.gnn_type,
                dropout=args.dropout,
            ).to(device)
            ema_model.load_state_dict(torch.load(prev_ema_path, map_location=device, weights_only=False))
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad = False
            print(f"[GNN] EMA model loaded from previous round: {prev_ema_path}")
        else:
            ema_model = copy.deepcopy(gnn_model)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad = False
            print(f"[GNN] EMA model initialized from current model (decay={args.ema_decay})")

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # Training loop
    best_eval_acc = best_test_acc = 0.0
    best_eval_mac_f1 = best_test_mac_f1 = 0.0
    counter = 0
    st_time = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, loss_orig, loss_pseudo = gnn_train_co_teaching(
            gnn_model, graph_data, optimizer,
            pseudo_mask, pseudo_y, args.alpha,
        )

        # Update EMA
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), gnn_model.parameters()):
                    ema_p.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

        [train_acc, val_acc, test_acc], [train_mac_f1, val_mac_f1, test_mac_f1], _ = \
            gnn_test(gnn_model, graph_data)

        if val_acc > best_eval_acc:
            best_eval_acc, best_test_acc = val_acc, test_acc
            best_eval_mac_f1, best_test_mac_f1 = val_mac_f1, test_mac_f1
            counter = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(gnn_model.state_dict(), args.save_path)
        else:
            counter += 1

        if epoch % args.print_freq == 0:
            pseudo_agree = evaluate_pseudo_label_acc(gnn_model, graph_data, pseudo_mask, pseudo_y)
            print(f"Epoch {epoch:03d}  loss={loss:.4f} (orig={loss_orig:.4f} pseudo={loss_pseudo:.4f})  "
                  f"train={train_acc:.1f} val={val_acc:.1f} test={test_acc:.1f}  "
                  f"pseudo_agree={pseudo_agree:.3f}")

        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = round(time.time() - st_time, 3)

    # Restore best model
    gnn_model.load_state_dict(torch.load(args.save_path, map_location=device))

    # Save EMA model and evaluate its accuracy
    ema_save_path = args.save_path.replace('.pt', '_ema.pt')
    ema_test_acc = 0.0
    ema_test_f1 = 0.0
    if ema_model is not None:
        torch.save(ema_model.state_dict(), ema_save_path)
        # Evaluate EMA model (observe only)
        [_, _, ema_test_acc], [_, _, ema_test_f1], _ = gnn_test(ema_model, graph_data)
        print(f"[EMA GNN] Test Acc {ema_test_acc:.2f}  Test Macro-F1 {ema_test_f1:.2f}")
        print(f"[EMA GNN] vs Best GNN: {ema_test_acc:.2f} vs {best_test_acc:.2f} "
              f"({'better' if ema_test_acc > best_test_acc else 'worse'})")

    print(f"\n[Final] Test Acc {best_test_acc:.2f}  Test Macro-F1 {best_test_mac_f1:.2f}  "
          f"Time {elapsed}s")
    print(f"Model saved to {args.save_path}")

    # Save metrics JSON for progress tracking
    metrics_path = args.save_path.replace('.pt', '_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            "test_acc": round(best_test_acc, 4),
            "test_macro_f1": round(best_test_mac_f1, 4),
            "val_acc": round(best_eval_acc, 4),
            "val_macro_f1": round(best_eval_mac_f1, 4),
            "ema_test_acc": round(ema_test_acc, 4),
            "ema_test_f1": round(ema_test_f1, 4),
            "epochs": epoch + 1,
            "time": elapsed,
        }, f, indent=2)
