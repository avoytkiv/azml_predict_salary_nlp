import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from common import DATA_DIR, RANDOM_STATE
from src.utils.logs import get_logger

def split_data(data_dir: str) -> None:
    logger = get_logger("DATA SPLIT", log_level="INFO")

    data_path = src_path / Path(data_dir) / "Train_rev1.csv.tar.gz"
    logger.info("Loading data from %s", data_dir)
    data = pd.read_csv(data_path, compression= "gzip", index_col=None)

    logger.info("Split data into train, validation and test sets in a 80/10/10 ratio")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=RANDOM_STATE)
    

    logger.info("Save train and test sets")
    train_data.to_csv(src_path / Path(data_dir) / "train.csv", index=False)
    val_data.to_csv(src_path / Path(data_dir) / "validation.csv", index=False)
    test_data.to_csv(src_path / Path(data_dir) / "test.csv", index=False)

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    split_data(**vars(args))

if __name__ == "__main__":
    main()
