import sys
import argparse
import logging
from pathlib import Path

import os
import mlflow
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from common import DATA_DIR, MODEL_DIR, TARGET_COLUMN, BATCH_SIZE, EPOCHS
from src.utils.logs import get_logger
from src.utils.train_utils import SalaryPredictor, BatchLoader


# def save_model(model_dir: str, model: nn.Module) -> None:
#     """
#     Saves the trained model.
#     """
#     input_schema = Schema(
#         [ColSpec(type="double", name=f"col_{i}") for i in range(784)])
#     output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
#     signature = ModelSignature(inputs=input_schema, outputs=output_schema)

#     code_paths = ["neural_network.py", "utils_train_nn.py"]
#     full_code_paths = [
#         Path(Path(__file__).parent, code_path) for code_path in code_paths
#     ]
#     shutil.rmtree(model_dir, ignore_errors=True)
#     logging.info("Saving model to %s", model_dir)
#     mlflow.pytorch.save_model(pytorch_model=model,
#                               path=model_dir,
#                               code_paths=full_code_paths,
#                               signature=signature)

def train(data_dir: str, model_dir: str, device: str, epochs=EPOCHS, batch_size=BATCH_SIZE) -> None:
    logger = get_logger("TRAIN", log_level="INFO")
    
    train_loader = BatchLoader(os.path.join(data_dir, "train_batches"))
    val_loader = BatchLoader(os.path.join(data_dir, "val_batches"))
    
    model = SalaryPredictor().to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        logger.info("Epoch %s/%s", epoch + 1, epochs)
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch[TARGET_COLUMN])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                loss = criterion(pred, batch[TARGET_COLUMN])
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Log Metrics with MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")


    logger.info("Plotting losses")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path(model_dir, "losses.png"))

    logger.info("Saving model")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()