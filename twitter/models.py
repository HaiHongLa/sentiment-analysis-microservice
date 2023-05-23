from django.db import models

class Tweet(models.Model):
    tweet_url = models.URLField()
    author_name = models.CharField(max_length=255)
    author_url = models.URLField()
    tweet_content = models.TextField()
    sentiment = models.CharField(max_length=20)
    last_updated_sentiment = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Tweet ID: {self.id}"