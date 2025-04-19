import torch
import torch.nn as nn


class RaagRecog(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, fc_dim, num_classes, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # shape: (batch, seq_len, embedding_dim)
        x, _ = self.lstm(x)    # shape: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len) for pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_dim)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
