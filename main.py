# Import Module
import os
import argparse
import numpy as np
import pandas as pd

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

# Import Custom
from model.bert import littleBert
from dataset import CustomDataset, CustomDataset_test, PadCollate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Data Loading...')
    start_time = time.time()

    data_path = './data/'
    data_list = glob(os.path.join(args.data_path, args.data_type))

if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type')