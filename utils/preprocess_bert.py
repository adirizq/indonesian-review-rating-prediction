import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from gensim.models import word2vec
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, max_len=100, batch_size=64, recreate=False):
        super(ReviewDataModule, self).__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        self.recreate = recreate

        self.dataset_dir = 'datasets'
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def load_data(self):
        dataset = pd.read_csv(f"{self.dataset_dir}/preprocessed_reviews.csv")
        data = dataset[['review_content', 'rating']]
        data = data.dropna()

        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for i, tr_d in enumerate(tqdm(data.values.tolist())):
            review = tr_d[0]
            label = tr_d[1]

            encoded_lbl = [0] * 5
            encoded_lbl[label-1] = 1

            tkn = self.tokenizer(text=review,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 truncation=True)

            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(encoded_lbl)

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)

        print('Splitting Data...')
        train_valid_len = round(len(tensor_dataset) * 0.8)
        test_len = len(tensor_dataset) - train_valid_len

        train_valid_data, test_data = torch.utils.data.random_split(
            tensor_dataset, [
                train_valid_len, test_len
            ]
        )

        train_len = round(len(train_valid_data) * 0.9)
        valid_len = len(train_valid_data) - train_len

        train_data, valid_data = torch.utils.data.random_split(
            train_valid_data, [
                train_len, valid_len
            ]
        )
        print('[Splitting Completed]\n')

        return train_data, valid_data, test_data

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=2
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=2
        )
