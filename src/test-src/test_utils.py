"""Provides functions to create loggers."""
import numpy as np
import logging
from typing import Text, Union
import sys
import os
import torch


TARGET_COLUMN = "Log1pSalary"


def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(
    name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG
) -> logging.Logger:
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string name or integer value
    Returns:
        logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger

class BatchLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.batch_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.pt')]
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.batch_files):
            raise StopIteration
        batch = torch.load(self.batch_files[self.index])
        self.index += 1
        return batch
    
    def __len__(self):
        return len(self.batch_files)
    
    def reset(self):
        self.index = 0

def eval_test(model, val_loader):
    """ 
    Let's make a prediction for all the validation data and plot the predictions against the true values
    The predictions are in log-scale, so we need to apply expm1 to them to get the actual salary values
    """
    
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch)
            val_preds.extend(pred)
            val_targets.extend(batch[TARGET_COLUMN])

    val_preds = np.expm1(val_preds)
    val_targets = np.expm1(val_targets)
    return val_preds, val_targets