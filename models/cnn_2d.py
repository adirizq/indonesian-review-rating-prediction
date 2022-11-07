import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class CNN2D(pl.LightningModule):
    def __init__(self,
                 word_embedding_weigth,
                 embedding_size,
                 out_channels=40,
                 window_sizes=[3, 4, 5],
                 num_classes=5,
                 learning_rate=1e-3,
                 dropout=0.5,
                 ) -> None:

        super(CNN2D, self).__init__()

        self.lr = learning_rate
        self.embedding_size = embedding_size
        self.output_dim = num_classes
        self.out_channels = out_channels

        weights = torch.FloatTensor(word_embedding_weigth)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.conv2d = nn.ModuleList([
            nn.Conv2d(1, out_channels, [window_size, embedding_size], padding=(window_size-1, 0)) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((out_channels*len(window_sizes)), num_classes)
        self.relu = nn.ReLU()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        embedding_out = self.embedding(x)

        prepared_conv_input = torch.unsqueeze(embedding_out, 1)

        out_conv = []

        for conv in self.conv2d:
            x = conv(prepared_conv_input)
            x = self.relu(x)
            x = x.squeeze(-1)
            x = F.max_pool1d(x, x.size(2))
            out_conv.append(x)

        logits = torch.cat(out_conv, 1)
        logits = logits.squeeze(-1)

        logits = self.dropout(logits)
        logits = self.linear(logits)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        reviews, targets = train_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        _, preds = torch.max(out.data, 1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'train loss': loss, 'train accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        reviews, targets = valid_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        _, preds = torch.max(out.data, 1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'validation loss': loss, 'validation accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, test_batch, batch_idx):
        reviews, targets = test_batch

        out = self(reviews)
        _, preds = torch.max(out.data, 1)

        return {"predictions": preds, "labels": targets}