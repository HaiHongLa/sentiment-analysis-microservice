from django.contrib import admin
from .models import RedditPost

@admin.register(RedditPost)
class RedditPostAdmin(admin.ModelAdmin):
    list_display = [field.name for field in RedditPost._meta.fields]
