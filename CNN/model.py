import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_weights,
                 num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.5, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data = torch.from_numpy(embedding_weights.astype(np.float32))
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))
            h = h.max(dim=2)[0]
            pooled.append(h)
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)
