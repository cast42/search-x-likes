import os
import random
import time
from collections.abc import Iterable
from typing import Any, cast

import openai
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from pydantic import BaseModel, Field
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

USER_NAME = "cast42"  # Replace with your Hugging Face username
REPO_NAME = "x_likes_queries"  # Change to your dataset name
NUM2PROCESS: int = 200  # Number tweets a query must be generated for
# Output file
OUTPUT_CSV: str = "search_queries_with_answers.csv"


# Define Pydantic model for structured output
class QuestionAnswerPair(BaseModel):
    query: str = Field(..., description="A search query relevant to the tweet text.")
    full_text: str = Field(
        ..., description="The full text of the retrieved tweet that is an asnwer relevant to the search query."
    )


class GPTResponse(BaseModel):
    queries: list[QuestionAnswerPair] = Field(..., description="A list of query-full text tweet pairs.")


class MissingTokenError(Exception):
    def __init__(self) -> None:
        super().__init__("HF_TOKEN is not found in the .env file.")


# Function to generate search queries with structured output
def generate_search_queries(client: openai.OpenAI, post: str) -> GPTResponse:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an assistant specialized in retrieval tasks. Your responses must be in valid JSON format.",
        },
        {
            "role": "user",
            "content": f"""
        The task is the following: given a the full text content of a tweet, you will have to generate queries that can be asked by a user to retrieve the tweet from a large documentary corpus.

        The query should be relevant to the full text of the tweet and should not be too specific or too general.
        The query should be about the subject in the full text of the tweet, and the answer needs to be found in the full text of the tweet.
        Queries are typical a list of keywords or topics the user is interested in.

        Generate a query that could be asked by a user without knowing the existence and the content of the list of tweets.

        Generate TWO pairs of questions and full text per tweet in a dictionary with the following format, your answer should ONLY contain this dictionary, NOTHING ELSE:
        {{
            "queries": [
                {{
                    "query": "XXXXXX",
                    "full_text": "YYYYYY"
                }},
                {{
                    "query": "XXXXXX",
                    "full_text": "YYYYYY"
                }}
            ]
        }}
        where XXXXXX is the query, YYYYYY is the full text of tweet that is retrieved when the user issues the query XXXXXX.

        Note: If there are no questions to ask about the full_text, return an empty list.
        Focus on making relevant queries concerning the full text of the tweet.

        Here is the full text of the tweet:
        "{post}"
        """,
        },
    ]

    # Cast to the union that parse(...) expects
    union_messages = cast(
        Iterable[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam],
        messages,
    )

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=union_messages,
            temperature=0.7,
            response_format=GPTResponse,
        )

        if (
            not response.choices[0].message.refusal
            and hasattr(response.choices[0].message, "parsed")
            and isinstance(response.choices[0].message.parsed, GPTResponse)
        ):
            result_json = response.choices[0].message.parsed
            return result_json
        else:
            return GPTResponse(queries=[])
    except Exception as e:
        print(f"Error processing tweet: {e}")
        return GPTResponse(queries=[])


def main() -> None:
    full_texts: list[str] = []
    queries: list[str] = []
    results_ids: list[str] = []

    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=API_KEY)

    # Load the 12,000 liked tweets
    dataset = load_dataset("cast42/x_likes", split="train")
    shuffled_dataset = dataset.shuffle(seed=1301)

    # Extract the 'full_text' column as a list of liked tweet texts
    liked_posts = shuffled_dataset["full_text"]
    ids = shuffled_dataset["tweet_id"]

    # Process tweets
    with Progress(
        TextColumn("[green]Generating queries..."),
        BarColumn(),  # Standard progress bar
        MofNCompleteColumn(),  # Shows X of Y complete
        TimeRemainingColumn(),  # Estimates remaining time
        transient=True,
    ) as progress:
        task = progress.add_task("", total=NUM2PROCESS)

        for post, id in zip(liked_posts[:NUM2PROCESS], ids[:NUM2PROCESS]):
            response_data = generate_search_queries(client, post)

            if response_data and response_data.queries:
                full_texts.append(post)
                queries.append(response_data.queries[0].query)
                results_ids.append(id)
                if len(response_data.queries) > 1:
                    full_texts.append(post)
                    queries.append(response_data.queries[1].query)
                    results_ids.append(id)

            time.sleep(random.uniform(0.001, 0.01))  # Rate limit handling
            progress.update(task, advance=1)
        progress.console.log(f"All {NUM2PROCESS} queries generated.")

    df = pd.DataFrame({"full_text": full_texts, "query": queries, "tweet_id": results_ids})
    df.to_csv(OUTPUT_CSV, index=False)

    print("Search queries with answers dataset saved as", OUTPUT_CSV)

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    # Get the Hugging Face token from the .env file
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise MissingTokenError()

    # Dataset path and repository details
    full_repo_name = f"{USER_NAME}/{REPO_NAME}"  # Format: username/repo_name

    hf_dataset.push_to_hub(full_repo_name, token=hf_token)
    print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{full_repo_name}")


if __name__ == "__main__":
    main()
