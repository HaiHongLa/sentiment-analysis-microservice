# Generated by Django 4.2.1 on 2023-05-24 00:56

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("reddit", "0002_redditpost_last_updated"),
    ]

    operations = [
        migrations.AlterField(
            model_name="redditpost",
            name="last_updated",
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name="redditpost",
            name="submission_date",
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
