from celery import shared_task
from django.utils import timezone
import time
from reddit.models import RedditPost
import os
import datetime

import sys
sys.path.append("../")
from text_classifier import classify_text_inputs


@shared_task
def update_posts():
    start = time.time()
    log_file = 'log.txt'
    error_file = 'error.txt'
    if not os.path.exists(log_file):
        open(log_file, 'w').close()  # Create an empty file
    if not os.path.exists(error_file):
        open(error_file, 'w').close()  # Create an empty file
    posts = RedditPost.objects.all()
    successes = 0
    failures = 0
    total = 0
    for post in posts:
        total += 1
        try:
            lastest_comments = post.get_updated_comments()
            if len(lastest_comments) == post.num_top_level_comments:
                successes += 1
                continue
            post.update()
            sa_results = classify_text_inputs(lastest_comments)
            post.set_num_top_level_comments(len(lastest_comments))
            post.set_sentiment_analysis_results(sa_results)
            post.save()
            successes += 1
        except Exception as e:
            failures += 1
            print(e)
            with open(error_file, 'a') as ef:
                f.write(str(e) + '\n')
            continue
    

    with open(log_file, 'a') as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"Updated {total} posts with {successes}/{total} successes and {failures}/{total} failures at {current_time}. Took {time.time() - start} seconds.\n")
