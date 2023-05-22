import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SMBackend.settings')
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
# Create the Celery application
app = Celery('SMBackend')

# Load the Celery configuration from Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover and include tasks from all registered Django apps
app.autodiscover_tasks()