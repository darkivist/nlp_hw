import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load the IMDb dataset from the CSV file
df = pd.read_csv('IMDB Dataset.csv')

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract text and labels
X_train, y_train = train_df['review'].values, (train_df['sentiment'] == 'positive').astype(int).values
X_test, y_test = test_df['review'].values, (test_df['sentiment'] == 'positive').astype(int).values

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
train_dataset = TensorDataset(X_train_padded, torch.tensor(y_train))
test_dataset = TensorDataset(X_test_padded, torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class TransformerMLP(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, transformer_dim, hidden_dim, num_layers, num_heads):
        super(TransformerMLP, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transformer_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # Output size for binary classification
        )

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded)
        transformer_output = transformer_output.mean(dim=1)  # Pooling the transformer output
        output = self.fc(transformer_output)
        return output.squeeze(1)

# Parameters for Transformer and MLP
vocab_size = vocab_size
embedding_dim = 128
transformer_dim = 128  # Should be the same as embedding_dim for this example
hidden_dim = 256
num_layers = 1
num_heads = 4

# Instantiate the TransformerMLP model
model = TransformerMLP(vocab_size, embedding_dim, transformer_dim, hidden_dim, num_layers, num_heads)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())  # Ensure labels are floats for BCEWithLogitsLoss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating - Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))  # Round the sigmoid output for binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy * 100:.2f}%")
