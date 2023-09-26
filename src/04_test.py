import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger
from src.utils.train_utils import SalaryPredictor

def load_test_data(data_dir: str) -> pd.DataFrame:
    pass

def test():
    """
    Tests the model on test data.
    """
    batch_size = 64
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    model = mlflow.pytorch.load_model(model_uri=model_dir)

    (test_loss, test_accuracy) = evaluate(device, test_dataloader, model,
                                          loss_fn)

    mlflow.log_param("test_loss", test_loss)
    mlflow.log_param("test_accuracy", test_accuracy)
    logging.info("Test loss: %f", test_loss)
    logging.info("Test accuracy: %f", test_accuracy)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test(**vars(args), device=device)


if __name__ == "__main__":
    main()