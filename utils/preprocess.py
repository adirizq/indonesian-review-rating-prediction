import sys
import multiprocessing
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import os
import zipfile
import urllib

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from gensim.models import word2vec
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


class Word2VecDataModule(pl.LightningDataModule):
    def __init__(self, max_len=100, batch_size=128, recreate=False):
        super(Word2VecDataModule, self).__init__()
        self.max_len = 100 if max_len is None else max_len
        self.batch_size = 128 if batch_size is None else batch_size
        self.recreate = False if recreate is None else recreate

        self.dataset_dir = 'datasets'
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

        if not os.path.exists('models/w2v/idwiki_word2vec_200_new_lower.model'):
            url = 'https://github.com/adirizq/data/releases/download/w2v/w2v_weight.zip'
            filename = url.split('/')[-1]

            urllib.request.urlretrieve(url, filename)

            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('models/w2v')

    def load_data(self):
        dataset = pd.read_csv(f"{self.dataset_dir}/preprocessed_reviews.csv")
        data = dataset[['review_content', 'rating']]
        data = data.dropna()

        y = []

        print('Fixing Label...')
        for rating in tqdm(data['rating'], desc='Fixing Label'):
            y.append(rating-1)
        print('[Fixing Completed]\n')

        tokenizer = Tokenizer()
        fit_text = data['review_content']
        tokenizer.fit_on_texts(fit_text)
        sequences = tokenizer.texts_to_sequences(data['review_content'])
        list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

        print('Padding Data...')
        x = pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], maxlen=self.max_len, padding='pre')
        print('[Padding Completed]\n')

        x = torch.tensor(x)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x, y)

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

    def word_embedding(self):
        dataset = pd.read_csv(f"{self.dataset_dir}/preprocessed_reviews.csv")
        data = dataset[['review_content', 'rating']]
        data = data.dropna()

        tokenizer = Tokenizer()
        fit_text = data['review_content']
        tokenizer.fit_on_texts(fit_text)

        print('Loading Word2Vec Model...')
        w2v_model = word2vec.Word2Vec.load('models/w2v/idwiki_word2vec_200_new_lower.model')
        w2v_weights = w2v_model.wv
        w2v_vocab_size, embedding_size = w2v_weights.vectors.shape
        print('[Loading Completed]\n')

        jumlah_index = len(tokenizer.word_index) + 1

        embedding_matrix = np.zeros((jumlah_index, 200))
        print('Creating Embedding Matrix...')
        for word, i in tqdm(tokenizer.word_index.items(), desc='Creating W2V Weigth'):
            try:
                embedding_vector = w2v_weights[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 200)
        print('[Embedding Matrix Completed]\n')

        del (w2v_weights)

        return embedding_matrix, w2v_vocab_size, embedding_size

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )


class BERTDataModule(pl.LightningDataModule):
    def __init__(self, max_len=100, batch_size=32, recreate=False):
        super(BERTDataModule, self).__init__()
        self.max_len = 100 if max_len is None else max_len
        self.batch_size = 32 if batch_size is None else batch_size
        self.recreate = False if recreate is None else recreate

        self.dataset_dir = 'datasets'
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def load_data(self):
        dataset = pd.read_csv(f"{self.dataset_dir}/preprocessed_reviews.csv")
        data = dataset[['review_content', 'rating']]
        data = data.dropna()

        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        print('Tokenizing Data...')
        for i, tr_d in enumerate(tqdm(data.values.tolist(), desc='Tokenizing Data')):
            review = tr_d[0]

            tkn = self.tokenizer(text=review,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 truncation=True)

            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(tr_d[1]-1)
        print('[Tokenizing Completed]\n')

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
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
