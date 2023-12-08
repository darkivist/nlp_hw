#sources: class lecture code (lecture 8 and 9)
#code debugged in chatgpt

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn

# Load data
df = pd.read_csv('Train_ML.csv')  # Update this line with your file path

# Select 'title' and 'abstract' columns as input text data and target columns as labels
X = df[['TITLE', 'ABSTRACT']].apply(lambda x: ' '.join(x), axis=1).values
y = df[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple tokenizer function
def simple_tokenizer(text):
    return text.split()

# Tokenize and prepare data
max_length = 128

# Ensure X_train and X_test are lists of strings
X_train = list(X_train)
X_test = list(X_test)

# Tokenize inputs using the simple tokenizer
X_train_tokenized = [simple_tokenizer(text)[:max_length] for text in X_train]
X_test_tokenized = [simple_tokenizer(text)[:max_length] for text in X_test]

# Convert text tokens to indices
vocab = {word: idx + 1 for idx, word in enumerate(set(word for review in X_train_tokenized + X_test_tokenized for word in review))}
vocab_size = len(vocab) + 1  # Add 1 for the padding token

X_train_indices = [[vocab.get(word, 0) for word in review] for review in X_train_tokenized]
X_test_indices = [[vocab.get(word, 0) for word in review] for review in X_test_tokenized]

# Pad sequences to a fixed length
X_train_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(review) for review in X_train_indices], batch_first=True)
X_test_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(review) for review in X_test_indices], batch_first=True)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_padded, torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(X_test_padded, torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class TransformerMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, transformer_dim, num_heads, num_classes):
        super(TransformerMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * max_length, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded)
        flattened_output = transformer_output.view(transformer_output.size(0), -1)  # Flatten the output
        output = self.mlp(flattened_output)
        return torch.sigmoid(output)

# Parameters for Transformer and MLP
vocab_size = vocab_size
embedding_dim = 128
transformer_dim = 128
num_heads = 4
num_classes = 6  # Number of target labels

# Instantiate the TransformerMLP model
model = TransformerMLP(vocab_size, embedding_dim, transformer_dim, num_heads, num_classes)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()  # Binary cross-entropy loss for multi-label classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating - Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {avg_test_loss:.4f}")

torch.save(model.state_dict(), 'mlp.pt')

# Load the test set for prediction
test_df = pd.read_csv('Test_submission_netid.csv')  # Update with your file path
X_test_submission = test_df[['TITLE', 'ABSTRACT']].apply(lambda x: ' '.join(x), axis=1).values

# Tokenize and prepare the test data similar to the training data
X_test_submission_tokenized = [simple_tokenizer(text)[:max_length] for text in X_test_submission]
X_test_submission_indices = [[vocab.get(word, 0) for word in review] for review in X_test_submission_tokenized]
X_test_submission_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(review) for review in X_test_submission_indices], batch_first=True)

# Create a TensorDataset and DataLoader for the test set
submission_dataset = TensorDataset(X_test_submission_padded)
submission_loader = DataLoader(submission_dataset, batch_size=16)

# Load the trained model onto the GPU
model = TransformerMLP(vocab_size, embedding_dim, transformer_dim, num_heads, num_classes)
model.load_state_dict(torch.load('mlp.pt'))
model.to(device)
model.eval()

# Generate predictions on the test set
predictions = []
with torch.no_grad():
    for inputs in tqdm(submission_loader, desc="Generating Predictions"):
        inputs = inputs[0].to(device)  # Move inputs to the GPU

        outputs = model(inputs)
        predicted_labels = torch.round(outputs).cpu().numpy().astype(int)
        predictions.extend(predicted_labels)

# Add predictions to the test DataFrame
test_df['MLP_Computer_Science'] = [pred[0] for pred in predictions]
test_df['MLP_Physics'] = [pred[1] for pred in predictions]
test_df['MLP_Mathematics'] = [pred[2] for pred in predictions]
test_df['MLP_Statistics'] = [pred[3] for pred in predictions]
test_df['MLP_Quantitative_Biology'] = [pred[4] for pred in predictions]
test_df['MLP_Quantitative_Finance'] = [pred[5] for pred in predictions]

# Save the DataFrame with predictions to a new CSV file
test_df.to_csv('MLP_predictions.csv', index=False)
