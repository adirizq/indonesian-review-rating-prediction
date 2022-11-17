import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from transformers import BertModel


class BertCNN2D(pl.LightningModule):
    def __init__(self,
                 num_classes=5,
                 learning_rate=1e-3,
                 dropout=0.5,
                 embedding_size=768,
                 in_channels=8,
                 out_channels=40,
                 window_sizes=[3, 4, 5],
                 ) -> None:

        super(BertCNN2D, self).__init__()

        self.lr = learning_rate
        self.output_dim = num_classes

        self.bert = BertModel.from_pretrained("indolem/indobert-base-uncased", output_hidden_states=True)
        self.conv2d = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, [window_size, embedding_size], padding=(window_size-1, 0)) for window_size in window_sizes
        ])
        self.linear = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear((out_channels*len(window_sizes)), num_classes)
        self.relu = nn.ReLU()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        bert_hidden_state = bert_out[2]
        bert_hidden_state = torch.stack(bert_hidden_state, dim=1)
        prepared_conv_input = bert_hidden_state[:, -8:]

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
        logits = self.classifier(logits)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = train_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = valid_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, targets = test_batch

        out = self(x_input_ids, x_token_type_ids, x_attention_mask)
        loss = self.criterion(out, targets)
        preds = torch.argmax(out, dim=1)

        accuracy = Accuracy().to(device='cuda')(preds, targets).item()
        self.log_dict({'test_loss': loss, 'test_acc': accuracy}, prog_bar=True, on_epoch=True)

        return loss