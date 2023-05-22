from django.utils import timezone
from django.db import models
import praw
from praw.models import MoreComments
import datetime
from reddit.config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)


class RedditPost(models.Model):
    post_id = models.CharField(max_length=10)
    title = models.CharField(max_length=300)
    author = models.CharField(max_length=100)
    submission_date = models.DateTimeField()
    score = models.IntegerField()
    num_comments = models.IntegerField()
    num_top_level_comments = models.IntegerField()
    body = models.TextField()
    subreddit = models.CharField(max_length=100)
    thumbnail = models.URLField(max_length=200)
    permalink = models.URLField(max_length=200)
    sentiment = models.JSONField(null=True)
    last_updated = models.DateTimeField()

    def init_attributes(self, url):
        submission = reddit.submission(url=url)
        self.post_id = submission.id
        self.title = submission.title
        self.author = submission.author.name
        self.submission_date = timezone.make_aware(
            datetime.datetime.utcfromtimestamp(submission.created_utc), timezone.utc)
        self.score = submission.score
        self.num_comments = submission.num_comments
        self.body = submission.selftext
        self.subreddit = submission.subreddit.display_name
        self.thumbnail = submission.thumbnail
        self.permalink = submission.permalink
        self.comments = list()
        for comment in submission.comments:
            if isinstance(comment, MoreComments):
                continue
            self.comments.append(comment.body)
        self.num_top_level_comments = len(self.comments)
        self.last_updated = timezone.make_aware(datetime.datetime.now(), timezone.utc)

    def get_comments(self):
        return self.comments

    def set_sentiment_analysis_results(self, sentiment_analysis_results):
        self.sentiment = sentiment_analysis_results
    
    def set_num_top_level_comments(self, num_comments):
        self.num_top_level_comments = num_comments

    def update(self):
        submission = reddit.submission(id=self.post_id)
        self.title = submission.title
        self.author = submission.author.name
        self.body = submission.selftext
        self.score = submission.score
        self.num_comments = submission.num_comments
        self.comments = list()
        for comment in submission.comments:
            if isinstance(comment, MoreComments):
                continue
            self.comments.append(comment.body)
        self.num_top_level_comments = len(self.comments)
        self.last_updated = timezone.make_aware(datetime.datetime.now(), timezone.utc)

    def get_updated_comments(self):
        submission = reddit.submission(id=self.post_id)
        updated_comments = []
        for comment in submission.comments:
            if isinstance(comment, MoreComments):
                continue
            updated_comments.append(comment.body)
        return updated_comments

    def __str__(self):
        return (
            f"Post ID: {self.post_id}\n"
            f"Title: {self.title}\n"
            f"Author: {self.author}\n"
            f"Submission Date: {self.submission_date}\n"
            f"Score: {self.score}\n"
            f"Number of Comments: {self.num_comments}\n"
            f"Number of top-level Comments: {self.num_top_level_comments}"
            f"Body: {self.body}\n"
            f"Subreddit: {self.subreddit}\n"
            f"Thumbnail: {self.thumbnail}\n"
            f"Permalink: {self.permalink}\n"
            f"Sentiment Analysis: {self.sentiment}"
            f"Last updated: {self.last_updated}"
        )
