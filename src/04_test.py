import sys
import argparse
import logging
from pathlib import Path

import shutil
import os
import mlflow
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from common import DATA_DIR, MODEL_DIR, PLOTS_DIR
from src.utils.logs import get_logger
from src.utils.train_utils import BatchLoader, eval_test


def test(data_dir: str,  model_dir: str, plots_dir: str) -> None:
    logger = get_logger("TEST", log_level="INFO")
    
    model_dir = src_path / Path(model_dir)
    plots_dir = src_path / Path(plots_dir)
    configs_dir = src_path / Path("configs")
    # Load the model
    model = torch.load(model_dir / "data/model.pth")
    test_loader = BatchLoader(os.path.join(data_dir, src_path / Path(data_dir) / "batches_validation"))
    
    test_loader.reset() 
    val_preds, val_targets = eval_test(model, test_loader)

    logger.info("Plotting predictions")
    plt.scatter(val_targets, val_preds, alpha=0.1)
    plt.xlabel("True salary")
    plt.ylabel("Predicted salary")
    plt.plot([0, 100000], [0, 100000], color='red', linestyle='--')
    plt.xlim([0, 100000])
    plt.ylim([0, 100000])

    plot_name = plots_dir / Path("predictions.png")
    plt.savefig(plot_name)

    with open(configs_dir / Path("run_id.txt"), "r") as file:
        run_id = file.read().strip()

    with mlflow.start_run(run_id=run_id):

        # shutil.rmtree(plots_dir, ignore_errors=True)
        logger.info("Logging plots")
        mlflow.log_artifact(plot_name)



def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    parser.add_argument("--plots_dir", dest="plots_dir", default=PLOTS_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    test(**vars(args))


if __name__ == "__main__":
    main()