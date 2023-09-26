import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
from sklearn.feature_extraction import DictVectorizer

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from common import DATA_DIR, TARGET_COLUMN, CATEGORICAL_COLUMNS, MAX_LEN, BATCH_SIZE
from src.utils.logs import get_logger

def preprocess_text(text: str) -> List[str]:
    """ This function takes a text and returns a list of tokens"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = WordPunctTokenizer().tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def categorical_vectorizer_func(data : pd.DataFrame):
        """ This function takes a dataframe and returns a vectorizer for categorical features"""
        # top_companies, top_counts = zip(*Counter(data['Company']).most_common(100))
        # recognized_companies = set(top_companies)
        # mask = ~data['Company'].isin(recognized_companies)
        # data.loc[mask, 'Company'] = 'Other'
        categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
        categorical_vectorizer.fit(data[CATEGORICAL_COLUMNS].apply(dict, axis=1))

        return categorical_vectorizer

class DataFeaturizer:
    def __init__(self, vocab_path, device, categorical_features):
        self.vocab = self.load_vocab(vocab_path)
        self.device = device
        self.categorical_features = categorical_features
        
    def load_vocab(self, vocab_path):
        with open(vocab_path, "r") as f:
            return json.load(f)

    def as_matrix(self, sequences, max_len=None):
        """ Convert a list of tokens into a matrix with padding """
        if isinstance(sequences[0], str):
            sequences = list(map(str.split, sequences))
        
        UNK_IX, PAD_IX = map(self.vocab.get, ["UNK", "PAD"])
        max_len = min(max(map(len, sequences)), max_len or float('inf'))
        matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
        for i,seq in enumerate(sequences):
            row_ix = [self.vocab.get(word, UNK_IX) for word in seq[:max_len]]
            matrix[i, :len(row_ix)] = row_ix
        
        return matrix

    def to_tensors(self, batch, device)->dict:
        '''
        This function takes a batch of data and converts it to tensors

        :param batch: a dict with {'title' : int64[batch, title_max_len]
        :param device: 'cuda' or 'cpu'
        :returns: a dict with {'title' : int64[batch, title_max_len]
        '''
        batch_tensors = dict()
        for key, arr in batch.items():
            if key in ["FullDescription", "Title"]:
                batch_tensors[key] = torch.tensor(arr, device=device, dtype=torch.int64)
            else:
                batch_tensors[key] = torch.tensor(arr, device=device)
        return batch_tensors

    def make_batch(self, data, categorical_features, device, max_len=MAX_LEN, word_dropout=0):
        """
        Creates a keras-friendly dict from the batch data.
        :param word_dropout: replaces token index with UNK_IX with this probability
        :returns: a dict with {'title' : int64[batch, title_max_len]
        """
        batch = {}
        batch["Title"] = self.as_matrix(data["Title"].values, max_len)
        batch["FullDescription"] = self.as_matrix(data["FullDescription"].values, max_len)
        batch['Categorical'] = categorical_features
        
        if word_dropout != 0:
            batch["FullDescription"] = self.apply_word_dropout(batch["FullDescription"], 1. - word_dropout)
        
        if TARGET_COLUMN in data.columns:
            batch[TARGET_COLUMN] = data[TARGET_COLUMN].values
        
        return self.to_tensors(batch, device)


    def apply_word_dropout(self, matrix, keep_prop):
        """ Sets random words in the matrix to UNK_IX with the probability `1 - keep_prop`"""
        UNK_IX, PAD_IX = map(self.vocab.get, ["UNK", "PAD"])
        dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop])
        dropout_mask &= matrix != PAD_IX
        return np.choose(dropout_mask, [matrix, np.full_like(matrix, UNK_IX)])


    def iterate_minibatches(self, data, categorical_features, batch_size=256, shuffle=True, cycle=False, device=None):
        """ iterates minibatches of data in random order """
        while True:
            indices = np.arange(len(data))
            if shuffle:
                indices = np.random.permutation(indices)

            for start in range(0, len(indices), batch_size):
                batch = self.make_batch(data.iloc[indices[start : start + batch_size]], 
                                        categorical_features.iloc[indices[start : start + batch_size]],
                                        device=device)
                yield batch
            
            if not cycle: break
    

def preprocess(data_dir: str, data_type: str, device: str) -> None:
    logger = get_logger("DATA LOAD", log_level="INFO")

    logger.info("Current working directory: %s", Path.cwd())
    logger.info("src path: %s", src_path)

    data_path = src_path / Path(data_dir) / (data_type + ".csv")
    vocab_path = src_path / Path(data_dir) / "vocab.json"
    BATCHES_DIR = src_path / Path(data_dir) / ("batches_" + data_type)
    if not os.path.exists(BATCHES_DIR):
        os.makedirs(BATCHES_DIR)

    logger.info("Loading data from %s", data_dir)
    data = pd.read_csv(data_path, index_col=None)

    logger.info("Tranforming data")
    data['Log1pSalary'] = np.log1p(data['SalaryNormalized']).astype('float32')
    
    data[CATEGORICAL_COLUMNS] = data[CATEGORICAL_COLUMNS].fillna('NaN')

    logger.info("How many missing values are there in 'FullDescription' and 'Title' columns")
    logger.info(data["FullDescription"].isnull().sum())
    logger.info(data["Title"].isnull().sum())

    logger.info("Drop rows with missing values in 'FullDescription' and 'Title' columns")
    data = data.dropna(subset=["FullDescription", "Title"])

    logger.info("Tokenize and lemmatize 'FullDescription' and 'Title' columns")
    nltk.download('stopwords')
    nltk.download('wordnet')
    data['Title'] = data['Title'].apply(preprocess_text)
    data['FullDescription'] = data['FullDescription'].apply(preprocess_text)

    logger.info("Category column is a categorical feature")
    categorical_vectorizer = categorical_vectorizer_func(data)
    categorical_features = categorical_vectorizer.transform(data[CATEGORICAL_COLUMNS].apply(dict, axis=1))
    logger.info("Number of categorical features %s", len(categorical_vectorizer.vocabulary_))

    if data_type == "train":
        logger.info("Build a vocabulary on train data")
        logger.info("Count how many times each word appears in 'FullDescription' and 'Title' columns")
        token_counts = Counter()
        for text in data["Title"]:
            token_counts.update(text.split())
        for text in data["FullDescription"]:
            token_counts.update(text.split())
        logger.info("Number of unique words: %s", len(token_counts))

        logger.info("Get a list of all tokens that occur at least 10 times.")
        min_count = 10
        tokens = [t for t, c in token_counts.items() if c >= min_count]
        tokens = sorted(tokens, key=lambda x: -token_counts[x])
        logger.info("Number of tokens: %s", len(tokens))

        logger.info("Add a special tokens for unknown and empty words")
        UNK, PAD = "UNK", "PAD"
        tokens = [UNK, PAD] + tokens

        logger.info("Build an inverse token index")
        token_to_id = {t: i for i, t in enumerate(tokens)}

        logger.info("Save vocabulary to file")
        with open(vocab_path, "w") as vocab_file:
            json.dump(token_to_id, vocab_file)

    logger.info("Map text lines into neural network inputs")
    data.index = rang