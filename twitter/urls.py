from django.urls import path
from . import views

urlpatterns = [
    path('create-tweet/', views.create_tweet, name='create-tweet'),
    path('tweets/', views.get_all_tweets, name="get-all-tweets"),
    path('tweet/<str:status>/', views.get_tweet_by_status, name="get-tweet-by-status"),
    path('delete-tweet/<str:status>/', views.delete_tweet_by_status, name="delete-tweet-by-status")
]
