import os

import numpy as np
import openai
import pandas as pd
import textual.widgets as tw
from sklearn.metrics.pairwise import cosine_similarity
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Label

EMBEDDING_MODEL: str = "text-embedding-3-small"
PARQUET_PATH: str = "./data/liked_posts_embedded.parquet"  # name and location of the generated parquet file


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
    if not isinstance(df[embeddings_col].iloc[0], np.ndarray):
        raise EmbeddingColumnTypeError(embeddings_col)

    embeddings = np.vstack(df[embeddings_col].to_list())

    # Compute cosine similarities
    similarities = cosine_similarity(embeddings, search_embedding.reshape(1, -1)).flatten()

    # Add similarities as a new column
    df["similarity"] = similarities

    # Get top-k rows sorted by similarity in descending order
    top_k_df = df.nlargest(k, "similarity")

    return top_k_df


class InputApp(App):
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
        response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        # Extract the embedding vector from the response
        search_embedding: list[float] = response.data[0].embedding
        results_widget: tw.Markdown = self.query_one(tw.Markdown)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results = get_top_k_embeddings(df, "embeddings", np.array(search_embedding), k=5)

        # Retrieve the found documents and update the markdown
        docs = [f"❱ {result}" for result in results["full_text"].values]

        results_widget.update("\n\n".join(docs))


app = InputApp()

if __name__ == "__main__":
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    df = pd.read_parquet(PARQUET_PATH)
    df["embeddings"] = df["embeddings"].map(lambda x: np.array(x))
    app.run()
