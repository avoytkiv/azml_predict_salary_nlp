
import argparse
import logging
from pathlib import Path
import pandas as pd
import mlflow
import torch
from torch import nn
import torch.nn.functional as F

from train_utils import SalaryPredictor, BatchLoader, evaluate, get_logger


TARGET_COLUMN = "Log1pSalary"

def train(epochs: int, batches_train: str, batches_validation: str, features_dim: str, model_dir: str, device: str) -> None:

    # Start logging
    mlflow.start_run()
    
    logger = get_logger("TRAIN", log_level="INFO")

    train_loader = BatchLoader(batches_train)
    val_loader = BatchLoader(batches_validation)
    
    logger.info("Extract length of vocabulary and number of categorical features from features_dim.csv")
    features_dim = pd.read_csv(features_dim, index_col=0)
    n_tokens = features_dim["n_tokens"].values[0]
    n_cat_features = features_dim["n_cat_features"].values[0]
    logger.info(f"n_tokens: {n_tokens}, n_cat_features: {n_cat_features}")
    
    logger.info("Initialize model, criterion and optimizer")
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

    
    mlflow.pytorch.save_model(pytorch_model=model, path=model_dir)

    mlflow.end_run()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", default=5, type=int)
    parser.add_argument("--batches_train", dest="batches_train", type=str, default="data/batches/train/")
    parser.add_argument("--batches_validation", dest="batches_validation", type=str, default="data/batches/validation/")
    parser.add_argument("--features_dim", dest="features_dim", type=str, default="data/features_dim.csv", help="path to features_dim.csv")
    parser.add_argument("--model_dir", dest="model_dir", default="data/processed/")
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()