import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report


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

        self.criterion = torch.nn.CrossEntropyLoss()

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
        preds = torch.argmax(out, dim=1)

        report = classification_report(targets.cpu(), preds.cpu(), labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)

        try:
            acc = report['accuracy']
        except KeyError:
            acc = report['micro avg']['f1-score']

        self.log_dict({'train_loss': loss,
                       'train_acc': acc,
                       'train_f1_macro': report['macro avg']['f1-score'],
                       'train_f1_weighted': report['weighted avg']['f1-score'],
                       'train_f1_rating_1': report['0']['f1-score'],
                       'train_f1_rating_2': report['1']['f1-score'],
                       'train_f1_rating_3': report['2']['f1-score'],
                       'train_f1_rating_4': report['3']['f1-score'],
                       'train_f1_rating_5': report['4']['f1-score'],
                       }, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        reviews, targets = valid_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        report = classification_report(targets.cpu(), preds.cpu(), labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)

        try:
            acc = report['accuracy']
        except KeyError:
            acc = report['micro avg']['f1-score']

        self.log_dict({'val_loss': loss,
                       'val_acc': acc,
                       'val_f1_macro': report['macro avg']['f1-score'],
                       'val_f1_weighted': report['weighted avg']['f1-score'],
                       'val_f1_rating_1': report['0']['f1-score'],
                       'val_f1_rating_2': report['1']['f1-score'],
                       'val_f1_rating_3': report['2']['f1-score'],
                       'val_f1_rating_4': report['3']['f1-score'],
                       'val_f1_rating_5': report['4']['f1-score'],
                       }, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        reviews, targets = test_batch

        out = self(reviews)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        report = classification_report(targets.cpu(), preds.cpu(), labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)

        try:
            acc = report['accuracy']
        except KeyError:
            acc = report['micro avg']['f1-score']

        self.log_dict({'test_loss': loss,
                       'test_acc': acc,
                       'test_f1_macro': report['macro avg']['f1-score'],
                       'test_f1_weighted': report['weighted avg']['f1-score'],
                       'test_f1_rating_1': report['0']['f1-score'],
                       'test_f1_rating_2': report['1']['f1-score'],
                       'test_f1_rating_3': report['2']['f1-score'],
                       'test_f1_rating_4': report['3']['f1-score'],
                       'test_f1_rating_5': report['4']['f1-score'],
                       }, prog_bar=True, on_epoch=True)

        return loss
