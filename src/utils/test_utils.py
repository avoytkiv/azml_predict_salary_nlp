"""Provides functions to create loggers."""
import numpy as np
import torch


TARGET_COLUMN = "Log1pSalary"


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