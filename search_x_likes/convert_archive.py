# https://gist.github.com/deepfates/78c9515ec2c2f263d6a65a19dd10162d
# Code by @deepfates
# https://x.com/deepfates/status/1858234134264697231

import argparse
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Literal, Optional

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MediaFile:
    id: str
    content_type: str
    path: str
    metadata: dict[str, Any]


@dataclass
class Content:
    id: str
    text: str
    metadata: dict[str, Any]
    timestamp: str
    parent_id: Optional[str]
    media_files: list[dict[str, Any]]
    content_source: str


@dataclass
class Thread:
    id: str
    contents: list[Content]


@dataclass
class Message:
    role: Literal["assistant", "user"]
    content: str


# Data extraction functions
def clean_json_string(json_string: str) -> str:
    return re.sub(r"^window\.[^=]+=\s*", "", json_string.strip()).rstrip(";")


def process_file(file_path: str) -> list[dict[str, Any]]:
    try:
        with open(file_path, encoding="utf-8") as f:
            data = clean_json_string(f.read())
            results: list[dict[str, Any]] = json.loads(data)  # Explicitly type results
            return results
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return []


def extract_manifest(file_path: str) -> dict[str, Any]:
    try:
        with open(file_path, encoding="utf-8") as file:
            content = clean_json_string(file.read())
            results: dict[str, Any] = json.loads(content)  # Explicitly type results
            return results
    except json.JSONDecodeError:
        match = re.search(r"window\.__THAR_CONFIG\s*=\s*({.*})", content, re.DOTALL)
        if match:
            result: dict[str, Any] = json.loads(match.group(1))
            return result
        logger.exception(f"Could not parse __THAR_CONFIG in manifest file: {file_path}")
        raise
    except Exception:
        logger.exception(f"Error extracting manifest from {file_path}")
        raise


def get_media_files(tweet_id: str, media_folder: str) -> list[str]:
    try:
        all_files = os.listdir(media_folder)
    except Exception:
        logger.exception(f"Error getting media files for tweet_id {tweet_id}")
        return []
    else:
        media_files = [
            f for f in all_files if f.startswith(f"{tweet_id}-") and os.path.getsize(os.path.join(media_folder, f)) > 0
        ]
        return media_files


def get_media_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".mp4", ".mov"):
        return "video"
    elif ext in (".jpg", ".jpeg", ".png", ".gif"):
        return "photo"
    return "unknown"


def extract_content(item: dict[str, Any], content_source: str, media_folder: str) -> list[Content]:
    content_id: str = str(item.get("id")) or str(item.get("tweetId"))
    text: str = str(item.get("text")) or str(item.get("fullText")) or str(item.get("full_text"))

    media_files: list[str] = get_media_files(content_id, media_folder)
    media_file_objects = [
        {
            "id": f"{content_id}_{os.path.splitext(media_file)[0]}",
            "content_type": get_media_type(media_file),
            "path": os.path.join(media_folder, media_file),
            "metadata": {"parent_tweet": item, "media_info": item.get("extended_entities", {}).get("media", [])},
        }
        for media_file in media_files
    ]

    return [
        Content(
            id=content_id,
            text=text,
            metadata=item,
            timestamp=item.get("created_at", ""),
            parent_id=item.get("in_reply_to_status_id"),
            media_files=media_file_objects,
            content_source=content_source,
        )
    ]


ExtractorType = Callable[[Any, str], list[Content]]


def process_file_wrapper(args: tuple[str, dict[str, Any], str, str]) -> list[Content]:
    archive_path, file_info, extractor_name, media_folder = args
    file_path = os.path.join(archive_path, file_info["fileName"])
    file_data = process_file(file_path)
    extractor: ExtractorType = globals()[extractor_name]  # Get the extractor function by name
    return extractor(file_data, media_folder)


def extract_content_data(
    archive_path: str, file_info: dict[str, Any], extractor: Callable, media_folder: str
) -> list[Content]:
    try:
        result: list[Content] = extractor(file_info["data"], media_folder)
    except Exception:
        logger.exception(f"Error extracting data with {extractor.__name__}")
        return []
    else:
        return result


def extract_data(archive_path: str, type_info: dict[str, Any], extractor: Callable) -> list[Content]:
    media_folder = os.path.join(archive_path, "data", "tweets_media")
    contents = []
    extractor_name = extractor.__name__

    with ProcessPoolExecutor() as executor:
        args_list = [
            (archive_path, file_info, extractor_name, media_folder) for file_info in type_info.get("files", [])
        ]
        futures = [executor.submit(process_file_wrapper, args) for args in args_list]

        total_futures = len(futures)
        logger.info(f"Processing {total_futures} files with {extractor_name}")
        completed_count = 0

        for completed_count, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                contents.extend(result)
            if completed_count % 10 == 0 or completed_count == total_futures:
                logger.info(f"Processed {completed_count}/{total_futures} files")

    logger.info(f"Total {extractor_name} extracted: {len(contents)} from {len(type_info.get('files', []))} files")
    return contents


def extract_tweets(file_data: list[dict[str, Any]], media_folder: str) -> list[Content]:
    logger.info(f"Extracting tweets from {len(file_data)} items")
    contents = [
        content
        for tweet in file_data
        if "tweet" in tweet
        for content in extract_content(tweet["tweet"], "tweet", media_folder)
    ]
    logger.info(f"Extracted {len(contents)} tweet contents")
    return contents


def extract_likes(file_data: list[dict[str, Any]], media_folder: str) -> list[Content]:
    logger.info(f"Extracting likes from {len(file_data)} items")
    contents = [
        content
        for like in file_data
        if "like" in like
        for content in extract_content(like["like"], "like", media_folder)
    ]
    logger.info(f"Extracted {len(contents)} like contents")
    return contents


def extract_archive_data(archive_path: str) -> dict[str, list[Content]]:
    try:
        manifest_path = os.path.join(archive_path, "data", "manifest.js")
        manifest = extract_manifest(manifest_path)
        data_types = manifest.get("dataTypes", {})

        extractors = {
            "tweets": extract_tweets,
            "like": extract_likes,
            # Add more extractors as needed
        }

        response = {}
        for data_type, extractor in extractors.items():
            if data_type in data_types:
                response[data_type] = extract_data(archive_path, data_types[data_type], extractor)
    except Exception:
        logger.exception("Error occurred during data extraction")
        return {}
    else:
        return response


# Data transformation functions
def clean_text(text: str, entities: Optional[dict] = None) -> str:
    if entities:
        for url in entities.get("urls", []):
            short_url = url.get("url", "")
            expanded_url = url.get("expanded_url", "")
            if short_url and expanded_url:
                text = text.replace(short_url, expanded_url)

    text = re.sub(r"https://t.co/\w+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_all_tweets(data: dict[str, list[Content]]) -> dict[str, Content]:
    logger.info("Combining tweets and likes into all_tweets")
    all_tweets = {tweet.id: tweet for tweet in data.get("tweets", []) if tweet.id}
    logger.info(f"Added {len(data.get('tweets', []))} tweets to all_tweets")

    likes = data.get("like", [])
    for like in likes:
        if like.id:
            all_tweets[like.id] = like
        else:
            logger.warning("Like without id encountered and skipped.")
    logger.info(f"Added {len(likes)} likes to all_tweets")
    logger.info(f"Total {len(all_tweets)} tweets/likes in all_tweets")

    return all_tweets


def get_conversation_texts(conversation: list[Content]) -> list[tuple[str, Literal["assistant", "user"]]]:
    return [
        (tweet.text, "assistant" if "full_text" in tweet.metadata else "user") for tweet in conversation if tweet.text
    ]


def trim_conversation_to_last_assistant(conversation_data: list[Message]) -> list[Message]:
    for i in range(len(conversation_data) - 1, -1, -1):
        if conversation_data[i].role == "assistant":
            return conversation_data[: i + 1]
    return []


def get_conversation_data(conversation: list[Content]) -> list[Message]:
    conversation_data: list[Message] = []
    current_role: Optional[Literal["assistant", "user"]] = None
    current_content: list[str] = []

    for text, role in get_conversation_texts(conversation):
        cleaned_text = clean_text(text)
        if cleaned_text:
            if role != current_role and current_role is not None:
                conversation_data.append(format_message(current_content, current_role))
                current_content = []
            current_role = role
            current_content.append(cleaned_text)

    if current_content and current_role in ("assistant", "user"):
        conversation_data.append(format_message(current_content, current_role))

    return trim_conversation_to_last_assistant(conversation_data)


def extract_threads_and_conversations(all_tweets: dict[str, Content]) -> tuple[list[Thread], list[list[Content]]]:
    """Extract threads and conversations from all tweets."""
    threads = []
    conversations = []

    # Keep track of processed tweet IDs to avoid duplicates
    processed_ids = set()

    for tweet in all_tweets.values():
        if tweet.id in processed_ids:
            continue

        if (
            tweet.content_source == "tweet"
            and tweet.parent_id
            and tweet.parent_id in all_tweets
            and not tweet.text.startswith("RT")
        ):
            # Initialize the chain
            chain = [tweet]
            current_tweet = tweet

            # Walk up the chain of replies
            while current_tweet.parent_id and current_tweet.parent_id in all_tweets:
                parent_tweet = all_tweets[current_tweet.parent_id]
                chain.append(parent_tweet)
                current_tweet = parent_tweet

                if current_tweet.id in processed_ids:
                    break  # Avoid cycles

            # Mark tweets as processed
            for t in chain:
                processed_ids.add(t.id)

            # Determine if it's a thread or conversation
            if all(t.content_source == "tweet" for t in chain):
                # This is a thread (user replying to themselves)
                threads.append(Thread(id=tweet.id, contents=list(reversed(chain))))
            else:
                # This is a conversation (user replying to others)
                conversations.append(list(reversed(chain)))

    return threads, conversations


# Data export functions
def process_media_files(media_files: list[dict[str, Any]], images_folder: str) -> list[str]:
    media_links = []
    for media_file in media_files:
        media_path = media_file.get("path")
        if media_path and os.path.isfile(media_path):
            orig_filename = os.path.basename(media_path)
            new_filename = f"_{orig_filename}"
            dest_path = os.path.join(images_folder, new_filename)
            shutil.copy(media_path, dest_path)
            media_links.append(f"![{new_filename}](images/{new_filename})")
        else:
            logger.warning(f"Invalid or missing media path: {media_path}")
    return media_links


def save_thread_markdown(thread: Thread, output_dir: str, media_folder: str, images_folder: str) -> None:
    if not thread.contents:
        logger.warning("Attempted to save an empty thread.")
        return

    try:
        date_str = thread.contents[0].timestamp
        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y").date()
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}")
        date = datetime.today().date()

    frontmatter = f"---\nDate: {date.isoformat()}\n---\n"

    thread_text = []
    for tweet in thread.contents:
        media_links = process_media_files(tweet.media_files, images_folder)
        cleaned_text = clean_text(tweet.text, tweet.metadata.get("entities"))
        combined_text = f"{cleaned_text}\n\n" + "\n\n".join(media_links)
        thread_text.append(combined_text)

    first_words = " ".join(thread_text[0].split()[:5])
    sanitized_filename = re.sub(r"[^\w\-_ ]", "", first_words).strip().replace(" ", "_")[:50]
    filename = f"{sanitized_filename}.md"
    file_path = os.path.join(output_dir, filename)

    top_tweet_id = thread.contents[0].id
    top_tweet_link = f"https://twitter.com/i/web/status/{top_tweet_id}"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{frontmatter}\n\n" + "\n\n".join(thread_text) + f"\n\n[View on Twitter]({top_tweet_link})")


def save_tweets_by_date(
    all_tweets: dict[str, Content], threads: list[Thread], output_dir: str, images_folder: str
) -> None:
    thread_ids = {tweet.id for thread in threads for tweet in thread.contents}
    non_thread_tweets = [
        tweet
        for tweet_id, tweet in all_tweets.items()
        if tweet_id not in thread_ids
        and not tweet.parent_id
        and tweet.content_source == "tweet"
        and not tweet.text.startswith("RT")
    ]

    tweets_by_date: dict[date, list[Content]] = {}
    for tweet in non_thread_tweets:
        date_str = tweet.timestamp
        if not date_str:
            logger.warning(f"Tweet missing date information: {tweet}")
            continue
        try:
            date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y").date()
            tweets_by_date.setdefault(date, []).append(tweet)
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")

    for date, tweets_on_date in tweets_by_date.items():
        filename = f"{date.isoformat()}.md"
        file_path = os.path.join(output_dir, filename)
        tweets_on_date.sort(key=lambda x: x.timestamp)
        content = "\n\n---\n\n".join(
            f"*{datetime.strptime(tweet.timestamp, '%a %b %d %H:%M:%S %z %Y').strftime('%I:%M %p')}*  \n{clean_text(tweet.text, tweet.metadata.get('entities'))}"
            + "".join(process_media_files(tweet.media_files, images_folder))
            for tweet in tweets_on_date
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def format_message(content: list[str], role: Literal["assistant", "user"]) -> Message:
    return Message(role=role, content="\n\n".join(content))


def format_conversation(conversation_data: list[Message], system_message: str) -> dict[str, Any]:
    messages = [{"role": "system", "content": system_message}]
    messages.extend([message.__dict__ for message in conversation_data])
    return {"messages": messages}


def save_conversations_to_jsonl(
    threads: list[Thread],
    conversations: list[list[Content]],
    output_path: str,
    system_message: str = "You have been uploaded to the internet",
) -> None:
    logger.info(f"Saving {len(conversations) + len(threads)} conversations to {output_path} in oai format")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for thread in threads:
            formatted_data = get_conversation_data(thread.contents)
            if not formatted_data:
                continue
            formatted_thread = format_conversation(formatted_data, system_message)
            f.write(json.dumps(formatted_thread) + "\n")

        for conversation in conversations:
            converstation_data = get_conversation_data(conversation)
            if not converstation_data:
                continue
            formatted_conv = format_conversation(converstation_data, system_message)
            f.write(json.dumps(formatted_conv) + "\n")


def main(archive_path: str, output_dir: str, output_formats: list[str], system_message: str) -> None:
    data = extract_archive_data(archive_path)
    all_tweets = get_all_tweets(data)
    threads, conversations = extract_threads_and_conversations(all_tweets)

    if "markdown" in output_formats:
        threads_output_dir = os.path.join(output_dir, "threads")
        images_folder = os.path.join(output_dir, "images")
        non_thread_output_dir = os.path.join(output_dir, "tweets_by_date")

        os.makedirs(threads_output_dir, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(non_thread_output_dir, exist_ok=True)

        logger.info(f"Saving {len(threads)} threads")
        for i, thread in enumerate(threads, start=1):
            save_thread_markdown(
                thread, threads_output_dir, os.path.join(archive_path, "data", "tweets_media"), images_folder
            )
            if i % 10 == 0 or i == len(threads):
                logger.info(f"Saved {i}/{len(threads)} threads")

        save_tweets_by_date(all_tweets, threads, non_thread_output_dir, images_folder)

    if "oai" in output_formats:
        output_path = os.path.join(output_dir, "conversations_oai.jsonl")
        save_conversations_to_jsonl(threads, conversations, output_path, system_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Twitter archive")
    parser.add_argument("--archive-path", default="test", help="Path to the Twitter archive directory")
    parser.add_argument("--output-dir", default="output", help="Directory where outputs will be saved")
    parser.add_argument(
        "--output-formats", nargs="+", default=["markdown", "oai"], help="Output formats to generate (markdown, oai)"
    )
    parser.add_argument(
        "--system-message", default="You have been uploaded to the internet", help="System message for the conversation"
    )
    args = parser.parse_args()

    main(args.archive_path, args.output_dir, args.output_formats, args.system_message)
