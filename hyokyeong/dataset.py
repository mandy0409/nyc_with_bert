# Import Module
import h5py
import numpy as np

# Import PyTorch
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.num_data = len(self.file[list(self.file.keys())[0]])

    def __getitem__(self, index):
        src = self.file[list(self.file.keys())[0]][index]
        hour = self.file[list(self.file.keys())[1]][index]
        weekday = self.file[list(self.file.keys())[3]][index]
        src_rev = np.flip(src)
        trg = self.file[list(self.file.keys())[4]][index]
        location = self.file[list(self.file.keys())[2]][index]
        return src, src_rev, hour, weekday, trg, location
    
    def __len__(self):
        return self.num_data


class Transpose_tensor:
    def __init__(self, dim=1):
        self.dim = dim

    def transpose_tensor(self, batch):
        (src, src_rev, weekday, hour, trg) = zip(*batch)
        batch_size = len(src)

        src_t = torch.FloatTensor(src)
        src_rev_t = torch.FloatTensor(src_rev)
        src_hour_t = torch.LongTensor(hour)
        src_weekday_t = torch.LongTensor(weekday)
        trg_t = torch.FloatTensor(trg)
        location_t = torch.FloatTensor(location)
        
        return src_t, src_rev_t, src_hour_t, src_weekday_t, trg_t, location_t

    def __call__(self, batch):
        return self.transpose_tensor(batch)

def getDataLoader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last):
    return DataLoader(dataset, drop_last=drop_last, batch_size=batch_size, 
                    collate_fn=Transpose_tensor(), shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers) 
