import contextlib
import random
from collections.abc import Generator
from time import perf_counter
from typing import TypedDict

import numpy as np
from sklearn.metrics import ndcg_score

from search_x_likes.list_likes_in_archive import load_likes

DATA_DIRECTORY: str = "data"
TOPK: int = 5

random.seed(1301)


class LikeInfo(TypedDict, total=False):
    tweetId: str
    fullText: str
    favoritedAt: str
    expandedUrl: str


@contextlib.contextmanager
def timer(subject: str = "time") -> Generator[None, None, None]:
    """Print the elapsed time. (Only used in debugging)"""
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    elapsed_ms = elapsed * 1000
    print(f"{subject} elapsed {elapsed_ms:.4f}ms")


# 1. Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(retrieved_docs: list[list[int]], ground_truths: list[int]) -> float:
    """
    Computes the Mean Reciprocal Rank (MRR) score.

    MRR measures how well the first relevant document is ranked among the retrieved documents.

    Args:
        retrieved_docs (List[List[int]]): A list of lists, where each inner list contains the ranked document indices for a query.
        ground_truths (List[int]): A list where each element represents the correct document index for a query.

    Returns:
        float: The MRR score, a value between 0 and 1.
    """
    ranks: list[float] = []
    for i, retrieved in enumerate(retrieved_docs):
        gt: int = ground_truths[i]
        rank = np.where(np.array(retrieved) == gt)[0]
        ranks.append(1 / (rank[0] + 1) if len(rank) > 0 else 0)
    return float(np.mean(ranks))


# 2. Recall@k
def recall_at_k(retrieved_docs: list[list[int]], ground_truths: list[int], k: int) -> float:
    """
    Computes Recall@k, which checks whether the correct document appears in the top-k results.

    Args:
        retrieved_docs (list[list[int]]): A list of lists containing ranked document indices for each query.
        ground_truths (list[int]): A list of the correct document indices for each query.
        k (int): The cutoff for the number of retrieved documents to consider.

    Returns:
        float: Recall@k score, a value between 0 and 1.
    """
    recall: list[int] = []
    for i, retrieved in enumerate(retrieved_docs):
        recall.append(1 if ground_truths[i] in retrieved[:k] else 0)
    return float(np.mean(recall))


# 3. NDCG@k
def ndcg_at_k(retrieved_docs: list[list[int]], ground_truths: list[int], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at rank k.

    NDCG rewards relevant documents appearing higher in the ranking.

    Args:
        retrieved_docs (list[list[int]]): A list of lists containing ranked document indices for each query.
        ground_truths (list[int]): A list of correct document indices for each query.
        k (int): The cutoff for the number of retrieved documents to consider.

    Returns:
        float: The NDCG@k score, a value between 0 and 1.
    """
    scores: list[float] = []
    for i, retrieved in enumerate(retrieved_docs):
        relevance: list[int] = [
            1 if retrieved[j] == ground_truths[i] else 0 for j in range(min(k, len(retrieved)))
        ]  # ensure not exceeding the length of retrieved
        scores.append(ndcg_score([relevance], [list(range(min(k, len(retrieved))))]))  # Sklearn NDCG calculation
    return float(np.mean(scores))


if __name__ == "__main__":
    likes: list[dict[str, LikeInfo]] = load_likes(DATA_DIRECTORY)
    posts = [like_obj.get("like", {}).get("fullText", "") for like_obj in likes]
    posts = [post for post in posts if len(post)]
    ground_truths = random.sample(range(len(posts)), min(200, len(posts)))
    random_secure = random.SystemRandom()
    queries = [random_secure.choice(posts[i].split()) for i in ground_truths if len(posts[i])]

    with timer("Retrieve relevant documents with exact search"):
        exact_search_results = []
        for query in queries:
            query_results = []
            number_of_matches: int = 0
            for idx, full_text in enumerate(posts):
                if query in full_text:
                    query_results.append(idx)
                    number_of_matches += 1
                    if number_of_matches > TOPK:
                        break
        exact_search_results.append(query_results)

    mrr = mean_reciprocal_rank(exact_search_results, ground_truths)
    recall_5 = recall_at_k(exact_search_results, ground_truths, k=5)
    ndcg_5 = ndcg_at_k(exact_search_results, ground_truths, k=5)

    # Print results
    print(f"MRR: {mrr:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    print(f"NDCG@5: {ndcg_5:.4f}")
    print(f"{ground_truths[0]=}")
    print(f"{queries[0]=}")
    print(f"{exact_search_results[0]=}")
