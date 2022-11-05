import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class LSTM(pl.LightningModule):
    def __init__(self,
                 word_embedding_weigth,
                 embedding_size,
                 hidden_size=128,
                 num_classes=5,
                 learning_rate=1e-3,
                 num_layers=3,
                 dropout=0.5,
                 ) -> None:

        super(LSTM, self).__init__()

        self.lr = learning_rate
        self.output_dim = num_classes
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        weights = torch.FloatTensor(word_embedding_weigth)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 5)
        self.sigmoid = nn.Sigmoid()

        self.criterion = torch.nn.BCELoss()

    def forward(self, x):

        embedding_out = self.embedding(x)
        embedding_out = self.dropout(embedding_out)
        lstm_out, _ = self.lstm(embedding_out)
        dropout_out = self.dropout(lstm_out)
        out = dropout_out[:, -1, :]
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        reviews, targets = train_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        _, preds = torch.max(out.data, 1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log("train accuracy", accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log("train loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        
        return loss

    def validation_step(self, valid_batch, batch_idx):
        reviews, targets = valid_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        _, preds = torch.max(out.data, 1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log("validation accuracy", accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log("validation loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)

        return loss

    def predict_step(self, test_batch, batch_idx):
        reviews, targets = test_batch

        out = self(reviews)
        _, preds = torch.max(out.data, 1)

        return {"predictions": preds, "labels": targets}

