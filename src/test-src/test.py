import argparse
import logging
from pathlib import Path

import os
import mlflow
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from test_utils import BatchLoader, get_logger, eval_test

TARGET_COLUMN = "Log1pSalary"

def test(data_dir: str,  model_dir: str, plots_dir: str) -> None:
    logger = get_logger("TEST", log_level="INFO")
    src_path = Path.cwd().parent
    
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
    parser.add_argument("--data_dir", dest="data_dir", default="data/")
    parser.add_argument("--model_dir", dest="model_dir", default="model/")
    parser.add_argument("--plots_dir", dest="plots_dir", default="plots/")
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    test(**vars(args))


if __name__ == "__main__":
    main()