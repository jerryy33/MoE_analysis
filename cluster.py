import os
import argparse
import random
import json
from collections import defaultdict

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer

from util import load_model


def cluster(embedding, d: int, k: int, seed: int):
    faiss.normalize_L2(embedding)
    kmeans = faiss.Kmeans(d, k, niter=100, verbose=True, spherical=True, seed=seed)
    kmeans.train(embedding)
    return kmeans.index.search(embedding, 1)[1]


def write_cluster_jsons(model_name: str, clusters: dict[str, np.ndarray], out_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id_to_token = {id_: tok for tok, id_ in tokenizer.get_vocab().items()}

    for name, cluster_ids in clusters.items():
        cluster_ids = cluster_ids.squeeze()
        cluster_map = defaultdict(list)

        for token_id, cluster_id in enumerate(cluster_ids):
            token = id_to_token.get(token_id, f"<UNK_{token_id}>")
            cluster_map[int(cluster_id)].append(token)

        filename = os.path.join(out_dir, f"{model_name.split('/')[1]}_K{name}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, indent=2, ensure_ascii=False)


@torch.inference_mode()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    model, _, _ = load_model(args.model_name, device_map="cpu")
    model_name = args.model_name.split("/")[1]

    embedding: np.ndarray = model.lm_head.weight.float().numpy()

    embedding = embedding - np.mean(embedding, axis=0)
    del model

    d = embedding.shape[1]

    cluster_ids = {}
    for k in args.ks:
        labels = cluster(embedding, d, k, args.seed)
        cluster_ids[f"{k}"] = labels

    write_cluster_jsons(args.model_name, cluster_ids, args.out_dir)
    filename = os.path.join(args.out_dir, f"{model_name}_labels.npz")
    np.savez(filename, **cluster_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster the output embedding matrix (unembedding) of a transformer model using spherical k-means.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model_name",
        default="allenai/OLMoE-1B-7B-0125",
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 50, 100, 1000, 5000],
        help="List of k values (number of clusters)",
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        default="./data/embed",
        help="Output directory for clustering results",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
