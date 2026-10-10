import asyncio
import json

import bm25s
import httpx2
import numpy as np
import openai
import pandas as pd
import pytest
import Stemmer
import torch
from datasets import Dataset
from model2vec import StaticModel
from sentence_transformers import util
from textual.widgets import Input, Markdown
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from search_x_likes import bm25_search, embed_posts, exact_search, generate_synthetic_eval_dataset
from search_x_likes.cosine_search import get_top_k_embeddings


def test_bm25_search_with_updated_textual(monkeypatch):
    corpus = ["Python dependency updates", "Train travel", "Coffee beans", "Garden plants", "Mountain walks"]
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer.stemWords))
    monkeypatch.setattr(bm25_search, "likes", [{"like": {"fullText": text}} for text in corpus], raising=False)
    monkeypatch.setattr(bm25_search, "corpus", corpus, raising=False)
    monkeypatch.setattr(bm25_search, "stemmer", stemmer, raising=False)
    monkeypatch.setattr(bm25_search, "retriever", retriever, raising=False)

    async def search():
        app = bm25_search.InputApp()
        async with app.run_test() as pilot:
            app.query_one(Input).value = "Python"
            await pilot.pause()
            results = app.query_one(Markdown).source
            assert results.startswith("❱ Python dependency updates")

    asyncio.run(search())


def test_exact_search_with_updated_textual(monkeypatch):
    monkeypatch.setattr(
        exact_search,
        "likes",
        [{"like": {"fullText": "Python dependency updates", "expandedUrl": "https://example.com/post"}}],
        raising=False,
    )

    async def search():
        app = exact_search.InputApp()
        async with app.run_test() as pilot:
            app.query_one(Input).value = "Python"
            await pilot.pause()
            assert "**Python** dependency updates" in app.query_one(Markdown).source

    asyncio.run(search())


@pytest.mark.parametrize("engine", ["pyarrow", "fastparquet"])
def test_parquet_and_cosine_search(tmp_path, engine):
    frame = pd.DataFrame({"tweet_id": ["1", "2"], "embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    path = tmp_path / "embeddings.parquet"
    frame.to_parquet(path, engine=engine)
    restored = pd.read_parquet(path, engine=engine)
    results = get_top_k_embeddings(restored, "embeddings", np.array([1.0, 0.0]), k=1)
    assert results["tweet_id"].tolist() == ["1"]
    assert results["similarity"].iloc[0] == pytest.approx(1.0)
    dataset = Dataset.from_pandas(restored)
    assert dataset[0]["embeddings"] == [1.0, 0.0]


def test_local_model_inference():
    unknown_word = "[UNK]"
    tokenizer = Tokenizer(WordLevel({unknown_word: 0, "python": 1, "coffee": 2}, unk_token=unknown_word))
    tokenizer.pre_tokenizer = Whitespace()
    model = StaticModel(vectors=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), tokenizer=tokenizer)
    embeddings = model.encode(["python", "coffee"])
    similarities = util.cos_sim(torch.tensor(embeddings), torch.tensor(embeddings))
    assert similarities.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_updated_openai_client_embedding_request():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    api_key = "unused"
    with openai.OpenAI(api_key=api_key, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        embedding = embed_posts.get_embedding(client, "Python\nupdates", model="text-embedding-3-small")
    assert embedding == [0.25, -0.5]
    assert requests[0]["input"] == ["Python updates"]
    assert requests[0]["model"] == "text-embedding-3-small"


def test_updated_openai_client_propagates_api_errors():
    def respond(request):
        return httpx2.Response(400, json={"error": {"message": "Invalid input", "type": "invalid_request_error"}})

    api_key = "unused"
    with (
        openai.OpenAI(api_key=api_key, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client,
        pytest.raises(openai.BadRequestError),
    ):
        embed_posts.get_embedding(client, "")


def test_batch_embedding_writes_parquet(tmp_path, monkeypatch):
    posts = [
        {"like": {"tweetId": "1", "fullText": "Python dependency updates", "expandedUrl": "https://example.com/1"}}
    ]
    (tmp_path / "like0.js").write_text("window.YTD.like.part0 = " + json.dumps(posts), encoding="utf-8")
    output = tmp_path / "embeddings.parquet"
    monkeypatch.setattr(embed_posts, "DATA_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(embed_posts, "SAVE_PATH", str(output))

    def respond(request):
        assert json.loads(request.content)["input"] == ["Python dependency updates"]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    api_key = "unused"
    with openai.OpenAI(api_key=api_key, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        monkeypatch.setattr(embed_posts.openai, "OpenAI", lambda **kwargs: client)
        embed_posts.main()
    frame = pd.read_parquet(output)
    assert frame["tweet_id"].tolist() == ["1"]
    assert frame["embeddings"].iloc[0].tolist() == [0.25, -0.5]


def test_structured_query_generation_with_updated_openai():
    pair = {"query": "python updates", "full_text": "Python dependency updates"}

    def respond(request):
        assert json.loads(request.content)["response_format"]["type"] == "json_schema"
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps({"queries": [pair]})},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    api_key = "unused"
    with openai.OpenAI(api_key=api_key, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        result = generate_synthetic_eval_dataset.generate_search_queries(client, pair["full_text"])
    assert result.model_dump() == {"queries": [pair]}
