import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from common import DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.utils.logs import get_logger

def split_data(data_dir: str, processed_data_dir: str) -> None:
    logger = get_logger("DATA SPLIT", log_level="INFO")

    raw_data_path = src_path / Path(data_dir) / "Train_rev1.csv.tar.gz"
    processed_data_path = src_path / Path(processed_data_dir)
    logger.info("Loading data from %s", data_dir)
    data = pd.read_csv(raw_data_path, compression= "gzip", index_col=None)

    logger.info("Split data into train, validation and test sets in a 80/10/10 ratio")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=RANDOM_STATE)
    

    logger.info("Save train and test sets")
    train_data.to_csv(processed_data_path / "train.csv", index=False)
    val_data.to_csv(processed_data_path / "validation.csv", index=False)
    test_data.to_csv(processed_data_path / "test.csv", index=False)

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--processed_data_dir", dest="processed_data_dir", default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    split_data(**vars(args))

if __name__ == "__main__":
    main()
