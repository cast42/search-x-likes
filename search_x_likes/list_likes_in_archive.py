import json
from pathlib import Path
from typing import TypedDict


class LikeInfo(TypedDict, total=False):
    tweetId: str
    fullText: str
    favoritedAt: str
    expandedUrl: str


def load_likes(data_directory: str) -> list[dict[str, LikeInfo]]:
    """
    Load 'like' data from JavaScript files in a given directory.

    This function searches for files matching the pattern 'like*.js' within the specified
    data directory, extracts JSON data from each file, and aggregates the 'like' data into a list.

    Args:
        data_directory (str): The path to the directory containing the 'like*.js' files.

    Returns:
        list[dict[str, LikeInfo]]: A list of dictionaries containing 'like' information.

    Raises:
        Exception: If there is an error processing any of the files, the exception is caught,
                   an error message is printed, and the function continues processing remaining files.
    """
    likes: list[dict[str, LikeInfo]] = []
    data_path: Path = Path(data_directory)
    like_files = data_path.glob("like*.js")
    for file_path in like_files:
        with file_path.open("r", encoding="utf-8") as f:
            content: str = f.read()
            try:
                json_data: str = content[content.index("=") + 1 :].strip()
                like_part: list[dict[str, LikeInfo]] = json.loads(json_data)
                likes.extend(like_part)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return likes


def main() -> None:
    DATA_DIRECTORY = "/Users/lode/Downloads/data"  # Adjust this path if your data directory is elsewhere
    likes = load_likes(DATA_DIRECTORY)
    for like_obj in likes:
        like: LikeInfo = like_obj.get("like", {})
        tweet_id: str = like.get("tweetId", "N/A")
        full_text: str = like.get("fullText", "")
        favorited_at: str = like.get("favoritedAt", "N/A")
        expanded_url: str = like.get("expandedUrl", "N/A")

        print("Tweet ID:", tweet_id)
        print("Favorited At:", favorited_at)
        print("Text:", full_text)
        print("URL:", expanded_url)
        print("-" * 40)


if __name__ == "__main__":
    main()
