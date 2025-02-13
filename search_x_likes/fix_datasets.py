import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding


class MissingTokenError(Exception):
    def __init__(self) -> None:
        super().__init__("HF_TOKEN is not found in the .env file.")


# Load .env file
load_dotenv()


def remove_column(dset: Dataset) -> Dataset:  # type: ignore
    return dset.remove_columns(["embeddings"])


def main() -> None:
    # Get the Hugging Face token from the .env file
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise MissingTokenError()

    dataset: DatasetDict = load_dataset("cast42/x_likes")  # type: ignore
    # dataset.push_to_hub("cast42/x_likes_embeddings_text_embedding_3_small", token=hf_token)

    # print(
    #     "Dataset successfully uploaded to: https://huggingface.co/datasets/cast42/x_likes_embeddings_text_embedding_3_small"
    # )

    static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
    model = SentenceTransformer(modules=[static_embedding])

    def create_embeddings(batch: dict[str, Any]) -> dict[str, Any]:
        """Create embeddings for a batch of text chunks."""
        batch["embeddings"] = model.encode(batch["full_text"])
        return batch

    # Create dataset with chunks and generate embeddings
    embeddings_dataset = dataset.map(create_embeddings, batched=True)
    embeddings_dataset.push_to_hub("cast42/x_likes_embeddings_potion_base_8M", token=hf_token)

    print("Dataset successfully uploaded to: https://huggingface.co/datasets/cast42/x_likes_embeddings_potion_base_8M")

    # Remove the "embeddings" column
    dataset_no_embeddings: DatasetDict = DatasetDict({split: remove_column(dset) for split, dset in dataset.items()})  # type: ignore

    # Step 4: Push the modified dataset under a new name
    dataset_no_embeddings.push_to_hub("cast42/x_likes", token=hf_token)


if __name__ == "__main__":
    main()
