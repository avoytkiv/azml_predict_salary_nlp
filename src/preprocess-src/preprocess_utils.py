import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from typing import List
import torch
import logging
from typing import Text, Union
import sys

TARGET_COLUMN = "Log1pSalary"
CATEGORICAL_COLUMNS = ["Category", "ContractType", "ContractTime"]
MAX_LEN = 10

def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(
    name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG
) -> logging.Logger:
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string name or integer value
    Returns:
        logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger

def preprocess_text(text: str) -> List[str]:
    """ This function takes a text and returns a list of tokens"""
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = WordPunctTokenizer().tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def categorical_vectorizer_func(data : pd.DataFrame):
        """ This function takes a dataframe and returns a vectorizer for categorical features"""
        categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
        categorical_vectorizer.fit(data[CATEGORICAL_COLUMNS].apply(dict, axis=1))

        return categorical_vectorizer

class DataFeaturizer:
    def __init__(self, vocab, device, categorical_features):
        self.vocab = vocab 
        self.device = device
        self.categorical_features = categorical_features

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
                                        categorical_features[start : start + batch_size, :],
                                        device=device,
                                        max_len=MAX_LEN,
                                        word_dropout=0)
                yield batch
            
            if not cycle: break