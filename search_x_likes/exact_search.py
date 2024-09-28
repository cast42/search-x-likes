import re
from typing import TypedDict

import textual.widgets as tw
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Label

from search_x_likes.list_likes_in_archive import load_likes

DATA_DIRECTORY: str = "/Users/lode/Downloads/data"  # Adjust this path if your data directory is elsewhere


class LikeInfo(TypedDict, total=False):
    tweetId: str
    fullText: str
    favoritedAt: str
    expandedUrl: str


# import bm25s

# Sample corpus
CORPUS = [
    "A cat is a small domesticated carnivorous mammal.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "A bird is a feathered, winged, bipedal, warm-blooded, egg-laying, vertebrate animal.",
    "A fish is a cold-blooded animal that lives in water.",
    "Elephants are the largest existing land animals.",
]

# Create a BM25 retriever
# retriever = bm25s.BM25(corpus=CORPUS)
# retriever.index(bm25s.tokenize(CORPUS))


def highlight_query(text: str, query: str) -> str:
    """Wraps every occurrence of the query string in bold Markdown in the text."""
    # Escape special regex characters in the query to avoid issues
    query_escaped = re.escape(query)

    # Use re.sub to replace all occurrences of the query with bold Markdown
    highlighted_text = re.sub(f"({query_escaped})", r"**\1**", text, flags=re.IGNORECASE)

    return highlighted_text


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
        yield Label("Enter search term.")
        yield Input(
            placeholder="Enter search term...",
        )
        # yield TextArea(id="results")  # Simplified TextArea
        yield tw.Markdown(markdown="Search results will be displayed here...")

    # Explicitly handle the changed event for the input widget
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change events."""
        query = event.value.strip()
        results_widget = self.query_one(tw.Markdown)
        search = []
        number_of_matches = 0
        for like_obj in likes:
            like: LikeInfo = like_obj.get("like", {})
            full_text: str = like.get("fullText", "")
            highlight_text: str = highlight_query(full_text, query)
            expanded_url: str = like.get("expandedUrl", "N/A")
            if query in highlight_text:
                search.append(f"❱ [{expanded_url}](expanded_url): " + highlight_text)
                number_of_matches += 1
                if number_of_matches > 5:
                    break

        results_widget.update("\n\n".join(search))


app = InputApp()

if __name__ == "__main__":
    likes: list[dict[str, LikeInfo]] = load_likes(DATA_DIRECTORY)
    app.run()
