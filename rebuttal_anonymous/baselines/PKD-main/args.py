import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="arxiv", help="cora, citeseer, pubmed, computers, photo")
    # masking
    parser.add_argument("--label_rate", type=int, default=5)  # 1, 2, 3
    return parser.parse_known_args()[0]
