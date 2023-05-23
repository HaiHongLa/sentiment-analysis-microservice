import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import time
import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and keep only alphanumeric and whitespace characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove unnecessary blank spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

with open('ml_models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = len(vectorizer.get_feature_names_out())
hidden_size = 128
num_classes = 2
model = TextClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('ml_models/text_classifier_model.pt'))


def classify_text_inputs(text_list):
    preprocessed_texts = [preprocess_text(text) for text in text_list]

    # Load the saved model
    model.eval()
    vectorized_text = vectorizer.transform(preprocessed_texts).toarray()

    # Convert to torch tensor
    inputs = torch.from_numpy(vectorized_text).float()

    # Make prediction
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        # _, predicted = torch.max(outputs.data, 1)
        # prediction = predicted.item()

    total_count = len(text_list)
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for probability in probabilities:
        if probability[0] > 0.7:
            negative_count += 1
        elif probability[1] > 0.7:
            positive_count += 1
        else:
            neutral_count += 1

    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100
    neutral_percentage = (neutral_count / total_count) * 100

    return {
        'positive_count': positive_count,
        'positive_percentage': positive_percentage,
        'negative_count': negative_count,
        'negative_percentage': negative_percentage,
        'neutral_count': neutral_count,
        'neutral_percentage': neutral_percentage,
        'analyzed_at': time.time(),
    }
