import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report


class CNN1D(pl.LightningModule):
    def __init__(self,
                 word_embedding_weigth,
                 embedding_size,
                 out_channels=40,
                 window_sizes=[3, 4, 5],
                 num_classes=5,
                 learning_rate=1e-3,
                 dropout=0.5,
                 ) -> None:

        super(CNN1D, self).__init__()

        self.lr = learning_rate
        self.embedding_size = embedding_size
        self.output_dim = num_classes
        self.out_channels = out_channels

        weights = torch.FloatTensor(word_embedding_weigth)
        self.embedding = nn.Embedding.from_pretrained(weights)

        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=window_size, padding=(window_size-1)) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((out_channels*len(window_sizes)), num_classes)
        self.relu = nn.ReLU()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        embedding_out = self.embedding(x)

        prepared_conv_input = embedding_out.permute(0, 2, 1)

        out_conv = []

        for conv in self.conv1d:
            x = conv(prepared_conv_input)
            x = self.relu(x)
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
