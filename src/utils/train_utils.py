"""Utilities that help with training neural networks."""

import os
import sys
from pathlib import Path  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from common import TARGET_COLUMN
from src.utils.logs import get_logger


def extract_feature_sizes(batch_dir: str) -> tuple[int, int]:
    """
    Extracts the feature sizes from saved batches.
    
    :param batch_dir: The directory containing the saved batches.
    :return: A tuple containing the sizes of tokens and categorical features.
    """
    n_tokens_set = set()
    n_cat_features_set = set()

    # Loop over all batch files in the specified directory
    for batch_file in os.listdir(batch_dir):
        if batch_file.endswith('.pt'):
            batch_path = os.path.join(batch_dir, batch_file)
            batch = torch.load(batch_path)
            
            # Extract and update the token and categorical feature sets
            n_tokens_set.update(batch['Title'].flatten().tolist(), batch['FullDescription'].flatten().tolist())
            n_cat_features_set.update(batch['Categorical'].flatten().tolist())
    
    # Return the sizes of the unique tokens and categorical features
    return len(n_tokens_set), len(n_cat_features_set)

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


class SalaryPredictor(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size=64):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, embedding_dim=hid_size) # embedding layer that converts token ids into embeddings
        
        # LSTM for title and description
        self.title_encoder = nn.LSTM(hid_size, hid_size, batch_first=True) 
        self.description_encoder = nn.LSTM(hid_size, hid_size, batch_first=True)
        
        # Fully-connected layer for categorical features
        self.fc_categorical = nn.Linear(n_cat_features, hid_size)
        
        # combining all three branches into one prediction
        self.fc_common = nn.Linear(hid_size * 3, hid_size) 
        self.fc_out = nn.Linear(hid_size, 1)  # output layer to predict salary
        
    def forward(self, batch):
        ''' 
        The input data for each branch is passed through the respective branch of the model, 
        and the output of each branch is combined and processed to produce the final output.
        '''
        title_emb = self.embed(batch['Title'])
        description_emb = self.embed(batch['FullDescription'])
        
        _, (title_hid, _) = self.title_encoder(title_emb)
        _, (description_hid, _) = self.description_encoder(description_emb)
        
        # For categorical
        categorical_hid = F.relu(self.fc_categorical(batch['Categorical']))
        
        # Concatenating all the features
        combined = torch.cat([title_hid.squeeze(0), description_hid.squeeze(0), categorical_hid], dim=1)
        
        common_hid = F.relu(self.fc_common(combined))
        out = self.fc_out(common_hid)

        return out.squeeze(-1)  # remove the last dimension
    
def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the given model for the whole dataset once.
    """
    assert len(val_loader) > 0, "Validation loader is empty!"
    model.eval()  
    val_loss = 0.0
    with torch.no_grad():  
        for batch in val_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                    
            pred = model(batch)
            loss = criterion(pred, batch[TARGET_COLUMN])
            val_loss += loss.item()
            
    val_loss /= len(val_loader)  
    return val_loss

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