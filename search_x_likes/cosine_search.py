# TUI application the calculates the cosine distance between the query and the texts of the posts in a parquet file.
# The location of the parquet file defined in constant PARQUET_PATH
# The parquetfile is created in the script embed_posts.py
# run with: uv run python search_x_likes/cosine_search.py
# Enter your search term in the TUI and hit enter

import contextlib
import logging
import os
from collections.abc import Generator
from time import perf_counter

import numpy as np
import openai
import pandas as pd
import pyarrow.parquet as pq
import textual.widgets as tw
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Label

EMBEDDING_MODEL: str = "text-embedding-3-small"
PARQUET_PATH: str = "./data/embeddings.parquet"  # name and location of the generated parquet file

"""
Timer context manager, only used in debug.
"""


@contextlib.contextmanager
def timer(subject: str = "time") -> Generator[None, None, None]:
    """Print the elapsed time. (Only used in debugging)"""
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    elapsed_ms = elapsed * 1000
    # log(f"{subject} elapsed {elapsed_ms:.4f}ms")
    logging.info(f"{subject} elapsed {elapsed_ms:.4f}ms")


class EmbeddingColumnTypeError(TypeError):
    def __init__(self, column_name: str):
        super().__init__(f"Column '{column_name}' must contain numpy arrays.")


# Function to retrieve top-k embeddings
def get_top_k_embeddings(df: pd.DataFrame, embeddings_col: str, search_embedding: np.ndarray, k: int) -> pd.DataFrame:
    """
    Retrieves the top-k most similar embeddings from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the embeddings.
        embeddings_col (str): Column name of embeddings.
        search_embedding (np.ndarray): The embedding of the search string.
        k (int): Number of top embeddings to retrieve.

    Returns:
        raise EmbeddingColumnTypeError(embeddings_col)
    """
    # Ensure the column contains numpy arrays
    # if not isinstance(df[embeddings_col].iloc[0], np.ndarray):
    #     print(df[embeddings_col].iloc[0])
    #     raise EmbeddingColumnTypeError(embeddings_col)

    embeddings = np.vstack(df[embeddings_col].to_list())

    # Compute cosine similarities
    with timer("Cosine similarity"):
        similarities = cosine_similarity(embeddings, search_embedding.reshape(1, -1)).flatten()

    # Add similarities as a new column
    df["similarity"] = similarities

    # Get top-k rows sorted by similarity in descending order
    top_k_df = df.nlargest(k, "similarity")

    return top_k_df


class InputApp(App):
    global df
    CSS = """
    Input {
        margin: 1 1;
    }
    Label {
        margin: 1 2;
    }
    TextArea {
        margin: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Set up the layout."""
        # Create the Input and TextArea widgets within a Vertical container
        yield Label(f"Search in {df.shape[0]} posts you liked on X.")
        yield Input(
            placeholder="Enter search term...",
        )
        # yield TextArea(id="results")  # Simplified TextArea
        yield tw.Markdown(markdown="Search results will be displayed here...")

    # Explicitly handle the changed event for the input widget
    # @on(Input.Changed)
    # def on_input_changed(self, event: Input.Changed) -> None:
    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission events (when Enter is pressed)."""
        query: str = event.value
        if len(query) < 4:
            return
        query = query.strip()
        with timer("Embed with openAI"):
            response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        # Extract the embedding vector from the response
        search_embedding: list[float] = response.data[0].embedding
        results_widget: tw.Markdown = self.query_one(tw.Markdown)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        with timer("get_top_k_embeddings"):
            results = get_top_k_embeddings(df, "embeddings", np.array(search_embedding), k=5)

        # Retrieve the found documents and update the markdown
        # docs = [f"❱ {result}" for result in results["full_text"].values]
        docs = [
            f"❱ [https://x.com/i/web/status/{tweet_id}](https://x.com/i/web/status/{tweet_id}) : {result}"
            for tweet_id, result in zip(results["tweet_id"].values, results["full_text"].values)
        ][::-1]

        results_widget.update("\n\n".join(docs))


app = InputApp()

if __name__ == "__main__":
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    # Configure logging to file
    logging.basicConfig(
        filename="textual_debug.log",  # Log file name
        filemode="w",  # Overwrite the file each time (use "a" to append)
        level=logging.INFO,  # Logging level (DEBUG for detailed logs)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    )
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    try:
        # Try to load the local parquet file
        # df = pd.read_parquet(PARQUET_PATH)
        # df["embeddings"] = df["embeddings"].map(lambda x: np.array(x))
        # df["embeddings"] = df["embeddings"].map(lambda x: np.array(ast.literal_eval(x)))
        table = pq.read_table(PARQUET_PATH)
        # Convert to Pandas DataFrame
        df = table.to_pandas()
    except FileNotFoundError as e:
        print(f"Error: Local file '{PARQUET_PATH}' not found. Attempting to load from Hugging Face dataset. {e}")
        try:
            # Fallback to loading the Hugging Face dataset
            dataset = load_dataset("cast42/x_likes", split="train")
            # Convert to Pandas DataFrame
            df = dataset.to_pandas()
        except Exception as hf_e:
            print(f"Failed to load from Hugging Face dataset: {hf_e}")
            raise  # Reraise the exception after logging
    except Exception as e:
        print(f"An unexpected error occurred while reading the local parquet file: {e}")
        raise  # Reraise the exception after logging

    app.run()
