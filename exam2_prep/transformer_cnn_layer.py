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


class TransformerCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_filters, filter_sizes, output_size, dropout):
        super(TransformerCNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=1
        )
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # Change dimensions for transformer
        transformer_output = self.transformer(embedded)
        transformer_output = transformer_output.permute(1, 2, 0)  # Reshape for conv1d input

        # Apply convolutions
        conv_outputs = [conv(transformer_output) for conv in self.conv_layers]
        pooled_outputs = [torch.nn.functional.max_pool1d(torch.nn.functional.relu(conv), conv.shape[2]).squeeze(2)
                          for conv in conv_outputs]
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))
        logits = self.fc(cat)
        return logits

# Parameters for Transformer and CNN
embedding_dim = 128
num_heads = 4
hidden_dim = 256
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 2  # Positive and negative classes
dropout = 0.5

# Instantiate the TransformerCNN model
transformer_cnn_model = TransformerCNN(
    vocab_size, embedding_dim, num_heads, hidden_dim, num_filters, filter_sizes, output_size, dropout
)

# Training loop
optimizer = torch.optim.Adam(transformer_cnn_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_cnn_model.to(device)

epochs = 3
for epoch in range(epochs):
    transformer_cnn_model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = transformer_cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    transformer_cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating - Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = transformer_cnn_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy * 100:.2f}%")
