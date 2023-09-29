
import argparse
import logging
from pathlib import Path
import shutil
import os
import mlflow
import torch
from torch import nn
import torch.nn.functional as F

from train_utils import SalaryPredictor, BatchLoader, evaluate, get_logger


TARGET_COLUMN = "Log1pSalary"

def train(data_dir: str,  model_dir: str, epochs: int, device: str) -> None:
    logger = get_logger("TRAIN", log_level="INFO")
    src_path = Path.cwd().parent

    data_dir = src_path / Path(data_dir)
    model_dir = src_path / Path(model_dir)
    configs_dir = src_path / Path("configs")
    
    train_loader = BatchLoader(os.path.join(data_dir / "batches_train"))
    val_loader = BatchLoader(os.path.join(data_dir / "batches_validation"))
    
    n_tokens, n_cat_features = 27449, 35# extract_feature_sizes(src_path / Path(data_dir) / "batches_train")
    logger.info(f"n_tokens: {n_tokens}, n_cat_features: {n_cat_features}")
    model = SalaryPredictor(n_tokens=n_tokens, n_cat_features=n_cat_features).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loader.reset()
        val_loader.reset()  
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch[TARGET_COLUMN])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # logger.info(f"Train loss: {loss.item()}")
        
        train_loss= train_loss / len(train_loader)
        train_losses.append(train_loss) 

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Log Metrics with MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    

    mlflow.end_run()
    with mlflow.start_run() as run:

        shutil.rmtree(model_dir, ignore_errors=True)
        logger.info("Saving model to %s", model_dir)
        mlflow.pytorch.save_model(pytorch_model=model, path=model_dir)

        run_id = run.info.run_id
    
    with open(configs_dir / Path("run_id.txt"), "w") as file:
        file.write(run_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="data/")
    parser.add_argument("--model_dir", dest="model_dir", default="data/processed/")
    parser.add_argument("--epochs", dest="epochs", default=5, type=int)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()