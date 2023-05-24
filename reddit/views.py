from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from reddit.models import RedditPost
from reddit.tasks import update_posts

import sys
sys.path.append("../")
from text_classifier import classify_text_inputs


def post_dict(post):
    return {
        "post_id": post.post_id,
        "title": post.title,
        "author": post.author,
        "submission_date": post.submission_date,
        "score": post.score,
        "num_comments": post.num_comments,
        "num_top_level_comments": post.num_top_level_comments,
        "body": post.body,
        "subreddit": post.subreddit,
        "thumbnail": post.thumbnail,
        "permalink": post.permalink,
        "sentiment": post.sentiment,
        "last_updated": post.last_updated,
    }


@api_view(['GET'])
def get_post_by_id(request, post_id):
    post = get_object_or_404(RedditPost, post_id=post_id)
    return JsonResponse({"post": post_dict(post)})


@api_view(['GET'])
def get_all_posts(request):
    posts = RedditPost.objects.all()
    post_list = [post_dict(post) for post in posts]
    return JsonResponse({"posts": post_list})


@api_view(['POST'])
def create_post(request):
    url = request.data['url']
    if 'reddit.com' not in url:
        return JsonResponse({'message': 'Error: not a valid Reddit URL'}, status=status.HTTP_400_BAD_REQUEST)
    post = RedditPost()
    post.init_attributes(url)
    post_id = post.post_id
    existing_post = RedditPost.objects.filter(post_id=post_id).first()
    if existing_post:
        return JsonResponse({'message': 'Error: This post already exists'}, status=status.HTTP_409_CONFLICT)
    sa_results = classify_text_inputs(post.get_comments())
    post.set_sentiment_analysis_results(sa_results)
    post.save()
    return JsonResponse({'message': 'Successfully created Reddit post', 'data': post_dict(post)})

@api_view(['PUT'])
def manual_update_post(request, post_id):
    post = get_object_or_404(RedditPost, post_id=post_id)
    lastest_comments = post.get_updated_comments()
    post.update()
    sa_results = classify_text_inputs(lastest_comments)
    post.set_num_top_level_comments(len(lastest_comments))
    post.set_sentiment_analysis_results(sa_results)
    post.save()
    return JsonResponse({"message": "Update successful", "data": post_dict(post)})

@api_view(['DELETE'])
def delete_post(request, post_id):
    post = get_object_or_404(RedditPost, post_id=post_id)
    post.delete()
    return JsonResponse({'message': 'Post deleted successfully'})

