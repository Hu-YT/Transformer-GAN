import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    def __init__(self, noise_dim, feature_dim, num_points, d_model, nhead=4, num_layers=6):
        super(TransformerGenerator, self).__init__()
        # Embedding layer to project input features to d_model dimension
        self.embedding = nn.Linear(noise_dim, d_model*num_points)
        self.positional_encoding = nn.Parameter(torch.randn(num_points, d_model))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers,
        )
        # Output layer
        self.output_layer = nn.Linear(d_model, 5)  # Output for generator (10*5)

    def forward(self, x):
        # x shape: (batch_size, noise_dim)
        x = self.embedding(x)  # Shape: (batch_size, num_points*d_model)
        x = x.view(x.size(0), 10, -1)
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2)  # Transformer expects shape: (num_points, batch_size, d_model)
        x = self.transformer(x)  # Shape: (num_points, batch_size, d_model)
        x = x.permute(1, 0, 2)  # Shape: (batch_size, num_points, input_dim)

        # x = self.fc(x)  # Shape: (batch_size, d_model)
        output = self.output_layer(x)  # Shape: (batch_size, num_points, feature_dim)
        # output = output.view(-1, seq_length, feature_dim)  # Reshape to (batch_size, seq_length, feature_dim)
        return output

class TransformerCritic(nn.Module):
    def __init__(self, feature_dim, num_points, d_model, nhead=4, num_layers=6):
        super(TransformerCritic, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(num_points, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(num_points * d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x).reshape(x.shape[0], -1)
        return output