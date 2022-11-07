import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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