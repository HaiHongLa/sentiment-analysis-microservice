# Generated by Django 4.2.1 on 2023-05-22 02:36

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("reddit", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="redditpost",
            name="last_updated",
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]