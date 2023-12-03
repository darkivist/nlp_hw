import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Custom Transformer model with MLP layer
class TransformerWithMLP(nn.Module):
    def __init__(self, input_dim, num_heads=8, ff_dim=512, num_blocks=4, mlp_hidden_dim=128, num_classes=10,
                 dropout_rate=0.1):
        super(TransformerWithMLP, self).__init__()

        self.positional_enc = nn.Embedding(input_dim, input_dim)

        # Define Transformer encoder blocks
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                    dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_blocks)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, x):
        # Add positional encoding
        positional_encoding = self.positional_enc(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        x += positional_encoding

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling over sequence length
        avg_pool = torch.mean(x, dim=1)

        # MLP layers
        output = self.mlp(avg_pool)
        return output


# Custom dataset for synthetic sequence data
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_length, input_dim):
        self.data = torch.randn(num_samples, seq_length, input_dim)
        self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define constants
input_dim = 64  # Dimensionality of input
num_classes = 2
num_samples = 1000
seq_length = 20
batch_size = 32
num_epochs = 5

# Create synthetic dataset
dataset = SyntheticDataset(num_samples=num_samples, seq_length=seq_length, input_dim=input_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the custom model
model = TransformerWithMLP(input_dim=input_dim, num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / (len(dataloader) * batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy * 100:.2f}%")
