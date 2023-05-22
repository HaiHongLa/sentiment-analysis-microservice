from django.urls import path
from . import views

urlpatterns = [
    path('create-post/', views.create_post, name='create_post'),
    path('post/<str:post_id>/', views.get_post_by_id, name='get_post_by_id'),
    path('posts/', views.get_all_posts, name='get_all_posts'),
    path('delete-post/<str:post_id>/', views.delete_post, name='delete_post'),
    path('manual-update-post/<str:post_id>/', views.manual_update_post, name='manual_update_posts')
]
