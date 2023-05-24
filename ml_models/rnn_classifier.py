import pandas as pd
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load the dataset
zf = zipfile.ZipFile("twitter_data.zip")
f = zf.open("training.1600000.processed.noemoticon.csv")
df = pd.read_csv(f, encoding='latin-1', names = ['target', 'id', 'time', 'query', 'username', 'content'])
df = df.iloc[795000:-795000]
zf.close()

# Remove punctuation and special characters
df['content'] = df['content'].str.replace('[^\w\s]','')

# Lowercase the text
df['content'] = df['content'].str.lower()

# Map target values to labels
label_map = {0: 0, 4: 1}
df["target"] = df["target"].map(label_map)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a custom dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, vectorizer):
        self.data = dataframe['content']
        self.target = dataframe['target']
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]
        vectorized_text = self.vectorizer.transform([text]).toarray().squeeze()
        target = self.target.iloc[index]
        return vectorized_text, target

# Create a count vectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data
vectorizer.fit(train_df['content'])

# Save the vectorizer model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Create the train and test datasets
train_dataset = TextDataset(train_df, vectorizer)
test_dataset = TextDataset(test_df, vectorizer)

# Define the model architecture
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

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the hyperparameters
input_size = len(vectorizer.get_feature_names_out())
hidden_size = 128
num_classes = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Initialize the model
model = TextClassifier(input_size, hidden_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs.float())
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

# Evaluate on test dataset
# Convert text data to vector representation
vectorized_text = vectorizer.transform(test_df['content']).toarray()

# Convert to torch tensor
inputs = torch.from_numpy(vectorized_text).float()

# Make predictions
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

# Calculate accuracy
total = predicted.size(0)
correct = (predicted == torch.tensor(test_df['target'].values)).sum().item()
accuracy = correct / total

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'text_classifier_model.pt')