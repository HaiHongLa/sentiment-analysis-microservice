# Sentiment Analysis Microservice (Development)

The Sentiment Analysis Microservice is a Django-based application currently under development. It aims to perform sentiment analysis on comments retrieved from social media platforms like Reddit and Twitter. This README provides an overview of the Reddit functionality.

## Features

- Retrieve posts from Reddit
- Analyze sentiments of comments using natural language processing techniques
- Store post data along with sentiment analysis results
- Continuous updates of post analysis using RabbitMQ and Celery

## Technologies Used

- Django: Python web framework for building the microservice
- Django REST framework: Facilitates the creation of RESTful APIs
- PRAW: Python wrapper for the Reddit API
- PyTorch: Deep learning library for text classification
- RabbitMQ: Message broker for asynchronous task processing
- Celery: Distributed task queue for executing tasks asynchronously

## Getting Started

### Prerequisites

- Python 3.7 or higher