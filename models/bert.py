import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report
from transformers import BertModel


class Bert(pl.LightningModule):
    def __init__(self,
                 num_classes=5,
                 learning_rate=1e-3,
                 dropout=0.5,
                 ) -> None:

        super(Bert, self).__init__()

        self.lr = learning_rate
        self.output_dim = num_classes

        self.bert = BertModel.from_pretrained("indolem/indobert-base-uncased")
        self.linear = nn.Linear(768, 512)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)
        self.tanh = nn.Tanh()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):

        _, cls_hs = self.bert(input_ids=input_ids, attention_mask=attention_mask).to_tuple()

        out = self.linear(cls_hs)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = train_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
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
        x_input_ids, x_token_type_ids, x_attention_mask, targets = valid_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
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
        x_input_ids, x_token_type_ids, x_attention_mask, targets = test_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
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
