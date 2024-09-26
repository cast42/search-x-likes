import sys
from io import StringIO

from search_x_likes.list_likes_in_archive import load_likes


def test_load_likes_no_files(tmp_path):
    """
    Test load_likes with no like*.js files in the directory.
    """
    likes = load_likes(tmp_path)
    assert likes == []


def test_load_likes_with_valid_file(tmp_path):
    """
    Test load_likes with a valid like*.js file.
    """
    # Create a valid like0.js file
    valid_content = """
    window.YTD.like.part0 = [
        {
            "like": {
                "tweetId": "1234567890",
                "fullText": "This is a test tweet",
                "favoritedAt": "2021-01-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/1234567890"
            }
        }
    ]
    """
    (tmp_path / "like0.js").write_text(valid_content, encoding="utf-8")

    # Call the function
    likes = load_likes(tmp_path)

    # Expected result
    expected_likes = [
        {
            "like": {
                "tweetId": "1234567890",
                "fullText": "This is a test tweet",
                "favoritedAt": "2021-01-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/1234567890",
            }
        }
    ]

    assert likes == expected_likes


def test_load_likes_with_invalid_file(tmp_path):
    """
    Test load_likes with an invalid like*.js file.
    """
    # Create an invalid like0.js file
    invalid_content = "window.YTD.like.part0 = invalid json content"
    (tmp_path / "like0.js").write_text(invalid_content, encoding="utf-8")

    # Capture the standard output
    captured_output = StringIO()
    sys.stdout = captured_output

    likes = load_likes(tmp_path)

    # Restore standard output
    sys.stdout = sys.__stdout__

    # No likes should be loaded
    assert likes == []

    # Check if error message was printed
    output = captured_output.getvalue()
    assert "Error processing file" in output


def test_load_likes_mixed_files(tmp_path):
    """
    Test load_likes with a mix of valid and invalid like*.js files.
    """
    # Valid file
    valid_content = """
    window.YTD.like.part0 = [
        {
            "like": {
                "tweetId": "1111111111",
                "fullText": "Valid tweet",
                "favoritedAt": "2021-02-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/1111111111"
            }
        }
    ]
    """
    (tmp_path / "like0.js").write_text(valid_content, encoding="utf-8")

    # Invalid file
    invalid_content = "window.YTD.like.part1 = invalid json content;"
    (tmp_path / "like1.js").write_text(invalid_content, encoding="utf-8")

    # Non-matching file
    (tmp_path / "other.js").write_text("This should not be read.", encoding="utf-8")

    # Capture the standard output
    captured_output = StringIO()
    sys.stdout = captured_output

    likes = load_likes(tmp_path)

    # Restore standard output
    sys.stdout = sys.__stdout__

    # Expected result
    expected_likes = [
        {
            "like": {
                "tweetId": "1111111111",
                "fullText": "Valid tweet",
                "favoritedAt": "2021-02-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/1111111111",
            }
        }
    ]

    assert likes == expected_likes

    # Check if error message was printed for the invalid file
    output = captured_output.getvalue()
    assert "Error processing file" in output


def test_load_likes_multiple_valid_files(tmp_path):
    """
    Test load_likes with multiple valid like*.js files.
    """
    # Create multiple valid files
    contents = [
        """
        window.YTD.like.part0 = [
            {
                "like": {
                    "tweetId": "2222222222",
                    "fullText": "First valid tweet",
                    "favoritedAt": "2021-03-01T00:00:00.000Z",
                    "expandedUrl": "https://twitter.com/test/status/2222222222"
                }
            }
        ]
        """,
        """
        window.YTD.like.part1 = [
            {
                "like": {
                    "tweetId": "3333333333",
                    "fullText": "Second valid tweet",
                    "favoritedAt": "2021-04-01T00:00:00.000Z",
                    "expandedUrl": "https://twitter.com/test/status/3333333333"
                }
            }
        ]
        """,
    ]
    for idx, content in enumerate(contents):
        (tmp_path / f"like{idx}.js").write_text(content, encoding="utf-8")

    likes = load_likes(tmp_path)

    # Expected result
    expected_likes = [
        {
            "like": {
                "tweetId": "3333333333",
                "fullText": "Second valid tweet",
                "favoritedAt": "2021-04-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/3333333333",
            }
        },
        {
            "like": {
                "tweetId": "2222222222",
                "fullText": "First valid tweet",
                "favoritedAt": "2021-03-01T00:00:00.000Z",
                "expandedUrl": "https://twitter.com/test/status/2222222222",
            }
        },
    ]

    assert likes == expected_likes
