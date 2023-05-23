from django.contrib import admin
from twitter.models import Tweet
# Register your models here.

@admin.register(Tweet)
class TweetAdmin(admin.ModelAdmin):
    list_display = [field.name for field in Tweet._meta.fields]