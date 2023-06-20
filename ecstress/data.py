import requests
import torch
from torch.utils.data import Dataset
import pandas as pd

class SlidingWindowDataset(Dataset):
    def __init__(self, ts, window_size, step_size):
        self.window_size = window_size
        self.ts = ts
        self.step_size = step_size
    
    def __len__(self):
        return (len(self.ts) - self.window_size + 1) // self.step_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(
            self.ts[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        )

def get_ett_dataset():
    r = requests.get(
        'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
    )
    open('ETTh1.csv', 'wb').write(r.content)
    ett = pd.read_csv('ETTh1.csv', index_col=0)
    ett.index = pd.to_datetime(ett.index)
    return ett
