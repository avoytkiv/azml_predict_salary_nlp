import argparse
import logging
from pathlib import Path
import mlflow
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from split_utils import get_logger

def split_data(raw_data: str, train_data: str, validation_data: str, test_data: str, random_state: int, test_train_ratio: float) -> None:
    # Start Logging
    mlflow.start_run()

    logger = get_logger("DATA SPLIT", log_level="INFO")
    logger.info("Print current working directory: %s", Path.cwd())
    
    data = pd.read_csv(raw_data, compression="gzip", index_col=None)

    logger.info("Split data into train, validation and test sets in a 80/10/10 ratio")
    train, test = train_test_split(data, test_size=test_train_ratio, random_state=random_state)
    test, val = train_test_split(test, test_size=0.5, random_state=random_state)

    mlflow.log_metric("train_samples", train.shape[0])
    mlflow.log_metric("validation_samples", val.shape[0])
    mlflow.log_metric("test_samples", test.shape[0])
    
    logger.info("Save train, validation and test sets")
    train.to_csv(train_data, index=False)
    val.to_csv(validation_data, index=False)
    test.to_csv(test_data, index=False)
    logger.info("Data saved")
 
    # Stop Logging
    mlflow.end_run()

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", dest="raw_data", default="data/raw/Train_rev1.csv.tar.gz")
    parser.add_argument("--train_data", dest="train_data", type=str, default="data/train/train.csv", help="path to train data")
    parser.add_argument("--validation_data", dest="validation_data", type=str, default="data/validation/validation.csv", help="path to validation data")
    parser.add_argument("--test_data", dest="test_data", type=str, default="data/test/test.csv", help="path to test data")
    parser.add_argument("--random_state", dest="random_state", type=int, default=42)
    parser.add_argument("--test_train_ratio", dest="test_train_ratio", type=float, required=False, default=0.2)
    args = parser.parse_args()
    logging.info(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    split_data(**vars(args))

if __name__ == "__main__":
    main()
