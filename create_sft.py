import json
import torch
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from common import DIRECT_PROMPTS
from common.dataloader import create_few_shot_dataset


def extract_title(text, max_len=120):
    """Extract title from ogbn-arxiv raw_text format.
    
    ogbn-arxiv format: 'title. abstract text #R##N# more abstract text'
    Strategy: take text before #R##N# separator, then cap at max_len.
    """
    if '#R##N#' in text:
        text = text.split('#R##N#')[0].strip()
    if len(text) > max_len:
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_test_samples', type=int, default=1000)
    parser.add_argument('--path_prefix', type=str, default=".")
    parser.add_argument('--use_neighbor_info', type=int, default=0,
                        help="Include neighbor text snippets in LLM prompt (0=off, 1=on)")
    
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return args

def create_sft_conversation(text: str, prompt: str, label: str, neighbor_ctx: str = "") -> Dict[str, List[Dict[str, str]]]:
    if neighbor_ctx:
        query_content = f"{text}\n{neighbor_ctx}\n{prompt}"
    else:
        query_content = f"{text}\n{prompt}"
    return {
        "conversations": [
            {"from": "human", "value": query_content},
            {"from": "gpt", "value": label}
        ]
    }

def create_unlabeled_dataset_from_test_mask(graph_data, dataset_name: str, output_path: str, suffix: str = "", use_neighbor_info: bool = False, adj=None, adj2=None) -> Tuple[List[Dict], Optional[str]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")
    
    all_test_indices = torch.where(graph_data.test_mask)[0].cpu().tolist()
    
    indices_to_use = all_test_indices
    
    node_ids_dir = os.path.dirname(output_path)
    os.makedirs(node_ids_dir, exist_ok=True)
    node_ids_filename = f"{dataset_name}{suffix}_unlabeled_node_ids.json"
    node_ids_path = os.path.join(node_ids_dir, node_ids_filename)
    
    with open(node_ids_path, 'w') as f:
        json.dump({"selected_node_ids": indices_to_use}, f, indent=2)
    print(f"Saved {len(indices_to_use)} unlabeled node IDs to {node_ids_path}")
    
    dataset = []
    for idx in indices_to_use:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]
        neighbor_ctx = get_neighbor_info(idx, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
        conversation = create_sft_conversation(origin_txt, prompt, true_label, neighbor_ctx)
        dataset.append(conversation)
    
    return dataset, node_ids_path


def create_train_dataset(graph_data, dataset_name: str, use_neighbor_info: bool = False, adj=None, adj2=None) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")   
    
    dataset = []

    train_indices = torch.where(graph_data.train_mask)[0].cpu().tolist()
    for idx in train_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]
        neighbor_ctx = get_neighbor_info(idx, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
        conversation = create_sft_conversation(origin_txt, prompt, true_label, neighbor_ctx)
        dataset.append(conversation)

    return dataset

def create_validation_dataset(graph_data, dataset_name: str, use_neighbor_info: bool = False, adj=None, adj2=None) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")

    dataset = []
    val_indices = torch.where(graph_data.val_mask)[0].cpu().tolist()

    for idx in val_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]
        neighbor_ctx = get_neighbor_info(idx, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
        conversation = create_sft_conversation(origin_txt, prompt, true_label, neighbor_ctx)
        dataset.append(conversation)

    return dataset

def create_test_dataset(graph_data, dataset_name: str, max_samples: int = -1, use_neighbor_info: bool = False, adj=None, adj2=None) -> List[Dict[str, Any]]:
    prompt = DIRECT_PROMPTS.get(dataset_name, "")
    if not prompt:
        raise ValueError(f"No prompt template found for dataset: {dataset_name}")

    dataset = []
    test_indices = torch.where(graph_data.test_mask)[0].cpu().tolist()

    if max_samples > 0 and len(test_indices) > max_samples:
        test_indices = random.sample(test_indices, max_samples)
        
    for idx in test_indices:
        origin_txt = graph_data.raw_texts[idx]
        true_label = graph_data.label_name[graph_data.y[idx].cpu().item()]
        neighbor_ctx = get_neighbor_info(idx, graph_data.raw_texts, adj, adj2) if use_neighbor_info and adj is not None else ""
        conversation = create_sft_conversation(origin_txt, prompt, true_label, neighbor_ctx)
        dataset.append(conversation)

    return dataset

def convert_to_sft_format(
    graph_data, 
    dataset_name: str, 
    output_path: str, 
    suffix: str = "",
    max_test_samples: int = -1,
    use_neighbor_info: bool = False,
):
    try:
        if suffix and not suffix.startswith('_'):
            suffix = f"_{suffix}"

        # Build adjacency cache once if needed
        adj, adj2 = None, None
        if use_neighbor_info:
            print("Building adjacency cache for neighbor info...")
            adj, adj2 = build_adj_cache(graph_data)
            print(f"  adj: {adj.shape}, nnz={adj.nnz}; adj2 nnz={adj2.nnz}")

        train_dataset = create_train_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name,
            use_neighbor_info=use_neighbor_info, adj=adj, adj2=adj2,
        )

        train_output_path = output_path.replace('.json', f'{suffix}_train.json')
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, ensure_ascii=False, indent=2)

        val_dataset = create_validation_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name,
            use_neighbor_info=use_neighbor_info, adj=adj, adj2=adj2,
        )

        if val_dataset:
            val_output_path = output_path.replace('.json', f'{suffix}_val.json')
            with open(val_output_path, 'w', encoding='utf-8') as f:
                json.dump(val_dataset, f, ensure_ascii=False, indent=2)

        test_dataset = create_test_dataset(
            graph_data=graph_data,
            dataset_name=dataset_name,
            max_samples=max_test_samples,
            use_neighbor_info=use_neighbor_info, adj=adj, adj2=adj2,
        )

        test_output_path = output_path.replace('.json', f'{suffix}_test.json')
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, ensure_ascii=False, indent=2)

        unlabeled_dataset, node_ids_path = create_unlabeled_dataset_from_test_mask(
            graph_data=graph_data,
            dataset_name=dataset_name,
            output_path=output_path,
            suffix=suffix,
            use_neighbor_info=use_neighbor_info, adj=adj, adj2=adj2,
        )
        unlabeled_output_path = output_path.replace('.json', f'{suffix}_unlabeled.json')
        with open(unlabeled_output_path, 'w', encoding='utf-8') as f:
            json.dump(unlabeled_dataset, f, ensure_ascii=False, indent=2)

        print(f"SFT splits: train={len(train_dataset)}, val={len(val_dataset)}, "
              f"test={len(test_dataset)}, unlabeled={len(unlabeled_dataset)}")

    except Exception as e:
        print(f"Error creating SFT datasets: {str(e)}")
        raise

def main():
    try:
        args = parse_arguments()
        seed = args.seed

        graph_data = create_few_shot_dataset(
            args.dataset,
            args.shots,
            args.seed,
            args.device,
            path_prefix=args.path_prefix
        )
        suffix = f"_{args.shots}_shot"

        convert_to_sft_format(
            graph_data=graph_data,
            dataset_name=args.dataset,
            output_path=args.output,
            suffix=suffix,
            max_test_samples=args.max_test_samples,
            use_neighbor_info=bool(args.use_neighbor_info),
        )

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())