import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
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

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.f1_weighted = MulticlassF1Score(num_classes=num_classes, average='weighted')
        self.f1_classes = MulticlassF1Score(num_classes=num_classes, average='none')

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

        acc = self.accuracy(preds, targets)
        f1_macro = self.f1_macro(preds, targets)
        f1_weighted = self.f1_weighted(preds, targets)
        f1_classes = self.f1_classes(preds, targets)

        self.log_dict({'train_loss': loss,
                       'train_acc': acc,
                       'train_f1_macro': f1_macro,
                       'train_f1_weighted': f1_weighted,
                       'train_f1_rating_1': f1_classes[0],
                       'train_f1_rating_2': f1_classes[1],
                       'train_f1_rating_3': f1_classes[2],
                       'train_f1_rating_4': f1_classes[3],
                       'train_f1_rating_5': f1_classes[4],
                       }, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = valid_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        acc = self.accuracy(preds, targets)
        f1_macro = self.f1_macro(preds, targets)
        f1_weighted = self.f1_weighted(preds, targets)
        f1_classes = self.f1_classes(preds, targets)

        self.log_dict({'val_loss': loss,
                       'val_acc': acc,
                       'val_f1_macro': f1_macro,
                       'val_f1_weighted': f1_weighted,
                       'val_f1_rating_1': f1_classes[0],
                       'val_f1_rating_2': f1_classes[1],
                       'val_f1_rating_3': f1_classes[2],
                       'val_f1_rating_4': f1_classes[3],
                       'val_f1_rating_5': f1_classes[4],
                       }, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = test_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        acc = self.accuracy(preds, targets)
        f1_macro = self.f1_macro(preds, targets)
        f1_weighted = self.f1_weighted(preds, targets)
        f1_classes = self.f1_classes(preds, targets)

        self.log_dict({'test_loss': loss,
                       'test_acc': acc,
                       'test_f1_macro': f1_macro,
                       'test_f1_weighted': f1_weighted,
                       'test_f1_rating_1': f1_classes[0],
                       'test_f1_rating_2': f1_classes[1],
                       'test_f1_rating_3': f1_classes[2],
                       'test_f1_rating_4': f1_classes[3],
                       'test_f1_rating_5': f1_classes[4],
                       }, on_epoch=True)

        return loss
