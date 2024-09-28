import httpx
from bs4 import BeautifulSoup


def get_tweet_date(url):
    """Gets the date of a tweet from a given URL."""

    # Send a GET request to the URL
    response = httpx.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the element containing the tweet's timestamp
    timestamp_element = soup.find("time", class_="css-901u3q")

    # Extract the date from the timestamp element
    date = timestamp_element.text

    return date


# Replace 'https://twitter.com/example/status/1234567890' with your actual tweet URL
tweet_url = "https://twitter.com/example/status/1234567890"
tweet_date = get_tweet_date(tweet_url)

print(tweet_date)
