from typing import TypedDict

import bm25s
import Stemmer  # optional: for stemming
import textual.widgets as tw
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Label

from search_x_likes.list_likes_in_archive import load_likes

DATA_DIRECTORY: str = "/Users/lode/Downloads/data"  # Adjust this path if your data directory is elsewhere
MAX_NUMBER_OF_MATCHES_SHOWN: int = 5


class LikeInfo(TypedDict, total=False):
    tweetId: str
    fullText: str
    favoritedAt: str
    expandedUrl: str


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
        yield Label(f"Search in {len(likes)} posts you liked on X.")
        yield Input(
            placeholder="Enter search term...",
        )
        # yield TextArea(id="results")  # Simplified TextArea
        yield tw.Markdown(markdown="Search results will be displayed here...")

    # Explicitly handle the changed event for the input widget
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change events."""
        query: str = event.value
        if len(query) < 4:
            return
        query = query.strip()
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        results_widget: tw.Markdown = self.query_one(tw.Markdown)
        if len(query_tokens) < 1:  # Do not retrieve when there are no tokens (e.g. word is a stopword)
            results_widget.update("")
            return

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=MAX_NUMBER_OF_MATCHES_SHOWN)

        # Retrieve the found documents and update the markdown
        docs = [f"❱ {results[0, i]}" for i in range(results.shape[1])]

        results_widget.update("\n\n".join(docs))


app = InputApp()

if __name__ == "__main__":
    stemmer = Stemmer.Stemmer("english")
    likes: list[dict[str, LikeInfo]] = load_likes(DATA_DIRECTORY)
    corpus = [like_obj.get("like", {}).get("fullText", "") for like_obj in likes]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    app.run()
