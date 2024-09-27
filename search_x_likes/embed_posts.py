import os  # for environment variables
from typing import TypedDict

import openai  # for generating embeddings
import pandas as pd

from search_x_likes.list_likes_in_archive import load_likes

DATA_DIRECTORY: str = "/Users/lode/Downloads/data"  # Adjust this path if your data directory is elsewhere
EMBEDDING_MODEL: str = "text-embedding-3-small"
SAVE_PATH: str = "./liked_posts_embedded.parquet"  # name and location of the generated parquet file


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
    response = client.embeddings.create(input=[cleaned_text], model=model)
    # Extract the embedding vector from the response
    embedding: list[float] = response.data[0].embedding
    return embedding


def main() -> None:
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    likes = load_likes(DATA_DIRECTORY)

    embedded_posts = []
    for like_obj in likes[:10]:
        like: LikeInfo = like_obj.get("like", {})
        tweet_id: str = like.get("tweetId", "N/A")
        full_text: str = like.get("fullText", "")
        expanded_url: str = like.get("expandedUrl", "N/A")
        embedding = get_embedding(client, full_text, EMBEDDING_MODEL)

        print("Tweet ID:", tweet_id)
        print("Text:", full_text)
        print("URL:", expanded_url)
        print("-" * 40)
        embedded_posts.append([tweet_id, full_text, expanded_url, embedding])

    df = pd.DataFrame(embedded_posts, columns=["tweet_id", "full_text", "expanded_url", "embedding"])

    df.to_parquet(SAVE_PATH, index=False)


if __name__ == "__main__":
    main()

# ideas: https://github.com/mneedham/LearnDataWithMark/blob/main/fts-vs-vector-search/fts_vector.ipynb
