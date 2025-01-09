import os  # for environment variables
from typing import TypedDict

import openai  # for generating embeddings
import pandas as pd
from rich.progress import Progress

from search_x_likes.list_likes_in_archive import load_likes

# DATA_DIRECTORY: str = "/Users/lode/Downloads/data"  # Adjust this path if your data directory is elsewhere
DATA_DIRECTORY: str = "/Users/lode/Downloads/twitter-2024-12-08-eb1fb01b92714ee7eb490e9622cb2b943d91a461b71f0cfa3e28f69b45424dfe/data"  # Adjust this path if your data directory is elsewhere
EMBEDDING_MODEL: str = "text-embedding-3-small"
SAVE_PATH: str = "./data/liked_posts_embedded.parquet"  # name and location of the generated parquet file


class LikeInfo(TypedDict, total=False):
    tweetId: str
    fullText: str
    favoritedAt: str
    expandedUrl: str


def get_embedding(client: openai.OpenAI, text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Generate embedding vectors for a text using the specified OpenAI model.

    Args:
        client (openai.openAI): An OpenAI client object
        text (str]): A string containing input text for which to generate embeddings.
        model (str, optional): The name of the embedding model to use.

    Returns:
        list[float]: A list of float representing an embedding vector.
    """
    cleaned_text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[cleaned_text], model=model)
        # Extract the embedding vector from the response
        embedding: list[float] = response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while generating the embedding: {e}")
        raise
    finally:
        if "embedding" not in locals():
            embedding = []
    return embedding


def main() -> None:
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    likes = load_likes(DATA_DIRECTORY)

    tweet_ids: list[str] = []
    full_texts: list[str] = []
    expanded_urls: list[str] = []
    for like_obj in likes:
        like: LikeInfo = like_obj.get("like", {})
        tweet_id: str = like.get("tweetId", "N/A")
        full_text: str = like.get("fullText", "")
        expanded_url: str = like.get("expandedUrl", "N/A")
        if len(full_text) < 5:
            continue
        tweet_ids.append(tweet_id)
        full_texts.append(full_text)
        expanded_urls.append(expanded_url)

    embeddings: list[list[float]] = []
    batch_size: int = 100
    with Progress() as progress:
        task = progress.add_task("[green]Generating embeddings...", total=len(full_texts))
        for i in range(0, len(full_texts), batch_size):
            batch = full_texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            response_dict = response.to_dict() if hasattr(response, "to_dict") else response
            if isinstance(response_dict, dict) and "data" in response_dict:
                embeddings.extend([item["embedding"] for item in response_dict["data"]])
            progress.update(task, advance=batch_size)

    data = {
        "tweet_id": tweet_ids,
        "full_text": full_texts,
        "expanded_url": expanded_urls,
        "embeddings": embeddings,
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    df.to_parquet(SAVE_PATH, index=False)


if __name__ == "__main__":
    main()

# ideas: https://github.com/mneedham/LearnDataWithMark/blob/main/fts-vs-vector-search/fts_vector.ipynb
