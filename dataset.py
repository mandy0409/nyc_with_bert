# Import Module
import os
import numpy as np
import pandas as pd

from glob import glob

# Import PyTorch
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type

        data_list = sorted(glob(os.path.join(data_path, data_type)))
        for i, data in enumerate(data_list):
            if i == 0:
                total_dat = data
            else:
                total_dat = pd.concat([total_dat, data])

        self.num_data = len(total_dat)

    def __getitem__(self, index):
        ix_dat = data.iloc[index:index+12] # 12 수정 필요
        weekday = ix_dat['pickup_weekday_index'].tolist()
        holiday = ix_dat['pickup_holiday'].tolist()
        time = #