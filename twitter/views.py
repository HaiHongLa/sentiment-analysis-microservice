from text_classifier import classify_text_inputs
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from twitter.models import Tweet

import sys
sys.path.append("../")


def tweet_to_dict(tweet):
    return {
        "id": tweet.id,
        "status": tweet.tweet_status,
        "tweet_url": tweet.tweet_url,
        "author_name": tweet.author_name,
        "author_url": tweet.author_url,
        "tweet_content": tweet.tweet_content,
        "sentiment": tweet.sentiment,
        "analyzed_at": tweet.analyzed_at,
    }

@api_view(['GET'])
def get_all_tweets(request):
    tweets = Tweet.objects.all()
    tweet_list = [tweet_to_dict(tweet) for tweet in tweets]
    return JsonResponse({"tweets": tweet_list})

@api_view(['GET'])
def get_tweet_by_status(request, status):
    tweet = get_object_or_404(Tweet, tweet_status=status)
    return JsonResponse({"tweet": tweet_to_dict(tweet)})


@api_view(['POST'])
def create_tweet(request):
    url = request.data['url']
    if 'twitter.com' not in url:
        return JsonResponse({'message': 'Error: not a valid Twitter URL'}, status=status.HTTP_400_BAD_REQUEST)
    existing_tweet = Tweet.objects.filter(tweet_url=url)
    if existing_tweet:
        return JsonResponse({'message': 'Error: This tweet already exists'}, status=status.HTTP_409_CONFLICT)
    tweet = Tweet()
    if not tweet.init_attributes(url):
        return JsonResponse({"message": "An error occurred when extracting the tweet. Please make sure your URL is correct(URL of a Tweet, containing https)"})
    sa_result = classify_text_inputs([tweet.tweet_content])
    tweet_sentiment = dict()
    if sa_result['positive_count'] == 1:
        tweet_sentiment['sentiment'] = 'positive'
    elif sa_result['negative_count'] == 1:
        tweet_sentiment['sentiment'] = 'negative'
    else:
        tweet_sentiment['sentiment'] = 'neutral'
    tweet_sentiment['analyzed_at'] = sa_result['analyzed_at']
    tweet.set_sentiment_analysis_results(tweet_sentiment)
    tweet.save()
    return JsonResponse({"message": "Created Tweet successfully", "data": tweet_to_dict(tweet)})

@api_view(['DELETE'])
def delete_tweet_by_status(request, status):
    tweet = get_object_or_404(Tweet, tweet_status=status)
    tweet.delete()
    return JsonResponse({'message': 'Tweet deleted successfully'})

