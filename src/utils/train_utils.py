"""Utilities that help with training neural networks."""

import os
import sys
from pathlib import Path  
import torch
import torch.nn as nn
import torch.nn.functional as F

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))


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
    
def evaluate():
    """
    Evaluates the given model for the whole dataset once.
    """
    train_losses = []
    val_losses = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count

    return (average_loss, accuracy)