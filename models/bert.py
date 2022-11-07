import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
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
        self.linear = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        self.tanh = nn.Tanh()

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = bert_out[0]

        pooler = hidden_state[:, 0]
        pooler = self.linear(pooler)
        pooler = self.tanh(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, y)
        preds = torch.argmax(out)
        targets = torch.argmax(y)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'train loss': loss, 'train accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, y)
        preds = torch.argmax(out)
        targets = torch.argmax(y)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'validation loss': loss, 'validation accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        preds = torch.argmax(out)
        targets = torch.argmax(y)

        return {"predictions": preds, "labels": targets}
