from django.db import models
from django.utils import timezone
import requests
from bs4 import BeautifulSoup
import re

def get_tweet_info(tweet_url):
    # Construct the oEmbed API URL
    oembed_url = f"https://publish.twitter.com/oembed?url={tweet_url}"

    # Send GET request to the oEmbed API
    response = requests.get(oembed_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant information from the response
        data = response.json()
        tweet_info = {
            "tweet_url": data.get("url"),
            "author_name": data.get("author_name"),
            "author_url": data.get("author_url"),
            "tweet_content": data.get("html")
        }
        return tweet_info
    else:
        print("Failed to retrieve tweet information.")
        return None

def extract_tweet_status(url):
    # Remove trailing slash if present
    if url.endswith('/'):
        url = url[:-1]

    # Split the URL by slashes and get the last element
    segments = url.split('/')
    tweet_id = segments[-1]

    return tweet_id


class Tweet(models.Model):
    tweet_status = models.CharField(max_length=50, default="None")
    tweet_url = models.URLField()
    author_name = models.CharField(max_length=255)
    author_url = models.URLField()
    tweet_content = models.TextField()
    sentiment = models.JSONField(default=dict)
    analyzed_at = models.DateTimeField(default=timezone.now)

    def init_attributes(self, url):
        tweet = get_tweet_info(url)
        if not tweet:
            return False
        self.tweet_status = extract_tweet_status(tweet['tweet_url'])
        self.tweet_url = tweet['tweet_url']
        self.author_name = tweet['author_name']
        self.author_url = tweet['author_url']
        tweet_text = tweet['tweet_content']
        soup = BeautifulSoup(tweet_text, 'html.parser')
        p_tags = soup.find_all('p')
        extracted_text = ' '.join([tag.get_text() for tag in p_tags])
        clean_text = re.sub(r'<[^>]+>', '', extracted_text)
        clean_text = re.sub(r'http\S+|www\S+', '', clean_text)  # Remove links
        self.tweet_content = clean_text
        return True

    def set_sentiment_analysis_results(self, sentiment_analysis_results):
        self.sentiment = sentiment_analysis_results

    def __str__(self):
        return f"Tweet ID: {self.id} by {self.author_name}"