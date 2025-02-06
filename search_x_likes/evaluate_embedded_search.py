import contextlib
import logging
from collections.abc import Generator
from time import perf_counter

import numpy as np
import torch
from datasets import load_dataset
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import ndcg_score

logging.basicConfig(
    filename="evaluate_embeddings.log",  # Log file name
    filemode="w",  # Overwrite the file each time (use "a" to append)
    level=logging.INFO,  # Logging level (DEBUG for detailed logs)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)


@contextlib.contextmanager
def timer(subject: str = "time") -> Generator[None, None, None]:
    """Print the elapsed time. (Only used in debugging)"""
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    elapsed_ms = elapsed * 1000
    # log(f"{subject} elapsed {elapsed_ms:.4f}ms")
    logging.info(f"{subject} elapsed {elapsed_ms:.4f}ms")


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # True if GPU is available
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if torch.backends.mps.is_available():
    print("Running on Apple Metal (MPS)")
else:
    print("MPS not available, running on CPU or CUDA")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

ds = load_dataset("cast42/x_likes_queries")

# embeddings_model, transformer_model = "sentence-transformers/all-MiniLM-L6-v2", True
# embeddings_model, transformer_model = "nomic-ai/modernbert-embed-base", True
# embeddings_model, transformer_model = "intfloat/multilingual-e5-large", True
# # https://bsky.app/profile/tomaarsen.com/post/3lgxsz227y22m
embeddings_model, transformer_model = "minishlab/potion-retrieval-32M", False
model = SentenceTransformer(embeddings_model) if transformer_model else StaticModel.from_pretrained(embeddings_model)

queries = ds["train"]["query"]
documents = ds["train"]["full_text"]

with timer(f"Embed with {embeddings_model} and retrieve on device {device}"):
    query_embeddings = model.encode(queries, normalize=True, convert_to_numpy=True)
    document_embeddings = model.encode(documents, normalize=True, convert_to_numpy=True)

    dot_scores = util.dot_score(query_embeddings, document_embeddings)  # [0]

    # Retrieve the top 5 results based on the dot product scores.
    top_results = torch.topk(dot_scores, k=15)
    torch.mps.synchronize()  # Ensure all operations are finished


# # Retrieve the top-k results
# top_k = 15  # Number of retrieved documents
retrieved_indices = top_results.indices  # Shape: (num_queries, k)

num_queries = retrieved_indices.shape[0]  # Get the number of queries

# Create ground truth indices as a tensor ranging from 0 to num_queries - 1
ground_truth_indices = torch.arange(num_queries)


### Compute Metrics ###


def mean_reciprocal_rank(retrieved_indices: torch.Tensor, ground_truth_indices: torch.Tensor) -> float:
    """Computes the Mean Reciprocal Rank (MRR) for a list of retrieved indices.

    Args:
        retrieved_indices (torch.Tensor): A tensor containing retrieved indices for each query.
        ground_truth_indices (torch.Tensor): A tensor containing ground truth indices for each query.

    Returns:
        float: The mean reciprocal rank score.
    """
    ranks: list[float] = []
    for i, retrieved in enumerate(retrieved_indices):
        gt: int | float = ground_truth_indices[i].item()  # Ground truth index
        rank = (retrieved == gt).nonzero(as_tuple=True)[0]
        ranks.append(1 / (rank[0].item() + 1) if len(rank) > 0 else 0)
    return float(np.mean(ranks))


def recall_at_k(retrieved_indices: torch.Tensor, ground_truth_indices: torch.Tensor, k: int) -> float:
    """Computes Recall@k, which measures the proportion of times the ground truth item
    is among the top-k retrieved items.

    Args:
        retrieved_indices (torch.Tensor): A tensor containing retrieved indices for each query.
        ground_truth_indices (torch.Tensor): A tensor containing ground truth indices for each query.
        k (int): The cutoff rank for computing recall.

    Returns:
        float: The recall@k score.
    """
    recall: list[int] = []
    for i, retrieved in enumerate(retrieved_indices):
        recall.append(1 if ground_truth_indices[i].item() in retrieved[:k] else 0)
    return float(np.mean(recall))


def ndcg_at_k(retrieved_indices: torch.Tensor, ground_truth_indices: torch.Tensor, k: int) -> float:
    """Computes the Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        retrieved_indices (torch.Tensor): A tensor containing retrieved indices for each query.
        ground_truth_indices (torch.Tensor): A tensor containing ground truth indices for each query.
        k (int): The cutoff rank for computing NDCG.

    Returns:
        float: The NDCG@k score.
    """
    scores: list[float] = []
    for i, retrieved in enumerate(retrieved_indices):
        relevance: list[int] = [1 if retrieved[j] == ground_truth_indices[i] else 0 for j in range(k)]
        scores.append(ndcg_score([relevance], [list(range(k))]))  # Sklearn NDCG
    return float(np.mean(scores))


# Compute scores
mrr: float = mean_reciprocal_rank(retrieved_indices, ground_truth_indices)
recall: float = recall_at_k(retrieved_indices, ground_truth_indices, k=5)  # Adjust k as needed
ndcg: float = ndcg_at_k(retrieved_indices, ground_truth_indices, k=5)  # Adjust k as needed

# Print results
print(f"Model: {embeddings_model} MRR: {mrr:.4f}, Recall@5: {recall:.4f}, NDCG@5: {ndcg:.4f}")
