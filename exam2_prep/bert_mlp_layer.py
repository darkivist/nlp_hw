import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Load the IMDb dataset from the CSV file
df = pd.read_csv('IMDB Dataset.csv')

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract text and labels
X_train, y_train = train_df['review'].values, (train_df['sentiment'] == 'positive').astype(int).values
X_test, y_test = test_df['review'].values, (test_df['sentiment'] == 'positive').astype(int).values

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and prepare data
max_length = 128

# Ensure X_train and X_test are lists of strings
X_train = list(X_train)
X_test = list(X_test)

# Tokenize inputs
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

train_dataset = TensorDataset(X_train_tokenized['input_ids'], X_train_tokenized['attention_mask'], torch.tensor(y_train))
test_dataset = TensorDataset(X_test_tokenized['input_ids'], X_test_tokenized['attention_mask'], torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize BERT
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False

# Define the MLP layer
class BERT_MLP(nn.Module):
    def __init__(self, bert_model, mlp):
        super(BERT_MLP, self).__init__()
        self.bert = bert_model
        self.mlp = mlp
        self.fc = nn.Linear(100, 2)  # Adjust output size according to your classification task

    def forward(self, input_ids, attention_mask):
        # Obtain BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = outputs.last_hidden_state

        # Flatten BERT embeddings
        flattened = bert_embeddings.view(bert_embeddings.size(0), -1)

        # Pass through MLP
        mlp_output = self.mlp(flattened)

        # Fully connected layer
        logits = self.fc(mlp_output)

        return logits

# Instantiate the MLP layer for BERT embeddings
mlp_layer = nn.Sequential(
    nn.Linear(bert_model.config.hidden_size * max_length, 100),  # Adjust input and output size of MLP
    nn.ReLU(),
    nn.Linear(100, 100)  # Adjust output size according to your architecture
)
model = BERT_MLP(bert_model, mlp_layer)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating - Epoch {epoch + 1}/{epochs}"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy * 100:.2f}%")