import os
import argparse
import logging
import numpy as np
import pandas as pd
from collections import Counter
import torch
import mlflow
import nltk

from preprocess_utils import preprocess_text, categorical_vectorizer_func, DataFeaturizer, get_logger

CATEGORICAL_COLUMNS = ["Category", "ContractType", "ContractTime"]
    
def preprocess(batch_size: int, train_data: str, validation_data: str, test_data: str, device: str, batches_train: str, batches_validation: str, batches_test: str) -> None:
    # Start Logging
    mlflow.start_run()

    logger = get_logger("DATA PROCESS", log_level="INFO")
    
    input_data_paths = [train_data, validation_data, test_data]
    output_batches_paths = [batches_train, batches_validation, batches_test]
    data_types = ["train", "validation", "test"]

    nltk.download('stopwords')
    nltk.download('wordnet')

    for data_path, batches_path, data_type in zip(input_data_paths, output_batches_paths, data_types):

        logger.info("Loading data from %s", data_path)
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
        
        data['Title'] = data['Title'].apply(preprocess_text)
        data['FullDescription'] = data['FullDescription'].apply(preprocess_text)

        logger.info("Category column is a categorical feature")
        categorical_vectorizer = categorical_vectorizer_func(data)
        categorical_features = categorical_vectorizer.transform(data[CATEGORICAL_COLUMNS].apply(dict, axis=1))
        logger.info("Number of categorical features %s", len(categorical_vectorizer.vocabulary_))
        mlflow.log_metric("categorical_features", len(categorical_vectorizer.vocabulary_))

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

            logger.info("Add a special tokens for unknown and empty words")
            UNK, PAD = "UNK", "PAD"
            tokens = [UNK, PAD] + tokens
            logger.info("Number of tokens: %s", len(tokens))
            mlflow.log_metric("tokens", len(tokens))

            logger.info("Build an inverse token index")
            vocab = {t: i for i, t in enumerate(tokens)}

        logger.info("Map text lines into neural network inputs")
        data.index = range(len(data))

        featurizer = DataFeaturizer(vocab, device, categorical_features)

        for idx, minibatch in enumerate(featurizer.iterate_minibatches(data, categorical_features, batch_size=batch_size, shuffle=True, device=device)):
            filename = os.path.join(batches_path, f"batch_{idx}.pt")
            torch.save(minibatch, filename)
        logger.info("Saved %s batches to %s", idx + 1, batches_path)
        mlflow.log_metric("batches_%s" % data_type, idx + 1)

# Stop Logging
mlflow.end_run()

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--train_data", dest="train_data", type=str, default="data/train/train.csv", help="path to train data")
    parser.add_argument("--validation_data", dest="validation_data", type=str, default="data/validation/validation.csv", help="path to validation data")
    parser.add_argument("--test_data", dest="test_data", type=str, default="data/test/test.csv", help="path to test data")
    parser.add_argument("--batches_train", dest="batches_train", type=str, default="data/batches/train/")
    parser.add_argument("--batches_validation", dest="batches_validation", type=str, default="data/batches/validation/")
    parser.add_argument("--batches_test", dest="batches_test", type=str, default="data/batches/test/")


    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess(**vars(args), device=device)


if __name__ == "__main__":
    main()

