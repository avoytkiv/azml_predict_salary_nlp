import argparse
import logging

import mlflow

from utils.train_utils import BatchLoader
from utils.test_utils import eval_test
from utils.logs import get_logger

TARGET_COLUMN = "Log1pSalary"

def test(model_dir: str, test_batches: str) -> None:
    # Start logging
    mlflow.start_run()

    logger = get_logger("TEST", log_level="INFO")
    
    logger.info("Loading model")
    model = mlflow.pytorch.load_model(model_dir)

    logger.info("Loading test data")
    test_loader = BatchLoader(test_batches)
    
    test_loader.reset() 
    val_preds, val_targets = eval_test(model, test_loader)
    logger.info(f"Test loss: {val_preds}")
    logger.info(f"Test loss: {val_targets}")

    # mlflow.log_metric("predicted_values", val_preds)
    # mlflow.log_metric("true_values", val_targets)

    # Plot predictions vs true values. Limit the range of the plot to 0-10K
    import matplotlib.pyplot as plt
    plt.scatter(val_targets, val_preds, alpha=0.1)
    plt.xlim(0, 100000)
    plt.ylim(0, 100000)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    # End logging
    mlflow.end_run()



def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", dest="model_dir", default="model/")
    parser.add_argument("--test_batches", dest="test_batches", type=str, default="data/batches/test/")
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    test(**vars(args))


if __name__ == "__main__":
    main()