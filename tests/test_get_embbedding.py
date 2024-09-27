import os

import openai
import pandas as pd
import pandas.testing as tm
import pytest

from search_x_likes.embed_posts import get_embedding


def test_empty_text():
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    with pytest.raises(openai.BadRequestError):
        get_embedding(client, "")


def test_invalid_model():
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    with pytest.raises(openai.NotFoundError):
        get_embedding(client, text="test", model="invalid_model")


def test_embedding_length(embedding_model: str = "text-embedding-3-small"):
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    text = "This is a test text."
    embedding = get_embedding(client, text, embedding_model)
    assert len(embedding) == 1536  # Adjust for the expected length of your model


def test_embedding_values(embedding_model: str = "text-embedding-3-small"):
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    text = "This is a test text."
    embedding = get_embedding(client, text, embedding_model)
    for value in embedding:
        assert -1 <= value <= 1


def test_consistency(embedding_model: str = "text-embedding-3-small"):
    api_key: str = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    text = "This is a test text."
    embedding1 = get_embedding(client, text, embedding_model)
    embedding2 = get_embedding(client, text, embedding_model)
    # Convert lists to pandas Series
    s1 = pd.Series(embedding1)
    s2 = pd.Series(embedding2)

    # Specify relative and absolute tolerance
    rtol = 1e-5  # Relative tolerance
    atol = 1e-8  # Absolute tolerance

    # Use assert_series_equal to check equality within tolerance
    tm.assert_series_equal(s1, s2, rtol=rtol, atol=atol)


# Add more test cases as needed, covering different scenarios and edge cases.

if __name__ == "__main__":
    pytest.main()
