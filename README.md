# search-x-likes

[![Release](https://img.shields.io/github/v/release/cast42/search-x-likes)](https://img.shields.io/github/v/release/cast42/search-x-likes)
[![Build status](https://img.shields.io/github/actions/workflow/status/cast42/search-x-likes/main.yml?branch=main)](https://github.com/cast42/search-x-likes/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/cast42/search-x-likes/branch/main/graph/badge.svg)](https://codecov.io/gh/cast42/search-x-likes)
[![Commit activity](https://img.shields.io/github/commit-activity/m/cast42/search-x-likes)](https://img.shields.io/github/commit-activity/m/cast42/search-x-likes)
[![License](https://img.shields.io/github/license/cast42/search-x-likes)](https://img.shields.io/github/license/cast42/search-x-likes)

Search posts from x that have liked yourself using the archive download files

- **Github repository**: <https://github.com/cast42/search-x-likes/>
- **Documentation** <https://cast42.github.io/search-x-likes/>
- **Pypy package** <https://pypi.org/project/search-x-likes/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:cast42/search-x-likes.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

### 5. Set OPENAI_API_KEY key

```bash
export OPENAI_API_KEY=<your key>
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/cast42/search-x-likes/settings/secrets/actions/new).
- Create a [new release](https://github.com/cast42/search-x-likes/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

## Development

Use `ruff` for linting and formatting, `mypy` for static code analysis, and `pytest` for testing.

The documentation is built with `mkdocs`, `mkdocs-material` and `mkdocstrings`.

## Search approaches

### Exact search

Retrieve the first k exact matches. This approach is implemented as a textual TUI in [search_x_likes/exact_search.py](search_x_likes/exact_search.py)

```python
k = 5 # retrieve k documents
retrieved = []
for idx, document in enumerate(documents):
    if query in document:
        retrieved.append(document)
    if idx > k:
        break
```

### BM25S

BM25S, an efficient Python-based
implementation of BM25 that only depends on
Numpy and Scipy. BM25S achieves up to a
500x speedup compared to the most popular
Python-based framework by eagerly computing BM25 scores during indexing and storing
them into sparse matrices.

This approach is implemented in [search_x_likes/bm25_search.py](search_x_likes/bm25_search.py)

[![Hugging Face Blog](https://img.shields.io/badge/Hugging%20Face-Blog-ffbe2f?logo=huggingface)](https://huggingface.co/blog/xhluca/bm25s) BM25 for Python: Achieving high performance while simplifying dependencies with BM25S⚡

[![arXiv](https://img.shields.io/badge/arXiv-2407.03618-b31b1b.svg)](https://arxiv.org/abs/2407.03618) Xing Han Lù, BM25S: Orders of magnitude faster lexical search via eager sparse scoring

[![GitHub](https://img.shields.io/badge/GitHub-xhluca%2Fbm25s-181717?logo=github)](https://github.com/xhluca/bm25s)

### Retrieve top-k documents scored with cosine similarity of their embeddings

Given the embedding of a document A and an embedding of a query B, score it's similarity as the normalized dot product
of the two vectors:

$$
\text{cosine similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
$$

This approach is implemented in [search_x_likes/cosine_search.py](search_x_likes/cosine_search.py)

### To evaluate the different retrieval methods, a synthetic dataset is created with llm

The code to generate the synthetic dataset with gpt-4o-mini is in [search_x_likes/generate_synthetic_eval_dataset.py](search_x_likes/generate_synthetic_eval_dataset.py)
It uses input the dataset that contains the post on x that are liked:

- [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffbe2f?logo=huggingface)](https://huggingface.co/datasets/cast42/x_likes) [cast42/x_likes](https://huggingface.co/datasets/cast42/x_likes)

- [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffbe2f?logo=huggingface)](https://huggingface.co/datasets/cast42/x_likes_queries) [cast42/x_likes_queries](https://huggingface.co/datasets/cast42/x_likes_queries)

The evaluation results are:

#### Retrieval Results (Colab CPU & GPU T4)

| Model                                  | MRR        | Recall@5   | NDCG@5     | Wall Time (CPU) | Wall Time (GPU) |
| -------------------------------------- | ---------- | ---------- | ---------- | --------------- | --------------- |
| BM25s                                  | **0.7414** | 0.8090     | 0.3249     | **51ms**        | -               |
| sentence-transformers/all-MiniLM-L6-v2 | 0.6517     | 0.9246     | 0.3964     | 20s             | 4.09s           |
| nomic-ai/modernbert-embed-base         | 0.6654     | **0.9472** | **0.4044** | 3m01s           | 6.82s           |
| intfloat/multilingual-e5-large         | 0.7063     | 0.9246     | 0.3823     | 7m57s           | 12.5s           |
| minishlab/potion-retrieval-32M         | 0.6346     | 0.8894     | 0.3813     | **2s**          | **1.64s**       |

## Contributing

All contributions are welcome, including more documentation, examples, code, and tests. Even questions.

## License - MIT

The package is open-sourced under the conditions of the [MIT license](https://choosealicense.com/licenses/mit/).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
