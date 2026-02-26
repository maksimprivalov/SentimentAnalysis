import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertSentiment(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased",
                 num_classes=2, dropout=0.3, freeze_layers=5):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        for p in self.distilbert.embeddings.parameters():
            p.requires_grad = False

        for i, layer in enumerate(self.distilbert.transformer.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_dim = self.distilbert.config.dim  # 768 for base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = output.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        return self.classifier(cls_hidden)
