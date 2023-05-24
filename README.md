# Sentiment Analysis Microservice (Development)

The Sentiment Analysis Microservice is a Django-based application currently under development. It aims to perform sentiment analysis on comments retrieved from social media platforms like Reddit and Twitter. This README provides an overview of the Reddit functionality.

## Features

- Retrieve posts from Reddit
- Analyze sentiments of comments using natural language processing techniques
- Store post data along with sentiment analysis results
- Continuous updates of post analysis using RabbitMQ and Celery
- Retrieve and analyze tweets from Twitter

## Technologies Used

- Django: Python web framework for building the microservice
- Django REST framework: Facilitates the creation of RESTful APIs
- PRAW: Python wrapper for the Reddit API
- PyTorch: Deep learning library for text classification
- RabbitMQ: Message broker for asynchronous task processing
- Celery: Distributed task queue for executing tasks asynchronously
- Beautiful Soup 4: A Python library for web scraping.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- RabbitMQ

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HaiHongLa/sentiment-analysis-microservice.git
   cd sentiment-analysis-microservice
   ```
2. Create a virtual environment and activate it:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the database:

    ```bash
    python manage.py migrate
    ```

5. Set up the RabbitMQ message broker:

    ```bash
    rabbitmq-server
    ```

6. Start the Celery worker:

    ```bash
    celery -A SMBackend worker --loglevel=info
    ```

7. Start the Django development server:

    ```bash
    python manage.py runserver
    ```

