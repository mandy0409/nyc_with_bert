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
    data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))
    #################################Fix##################################
    dataset_dict = {
        'train': CustomDataset(args.data_path, args.data_type),
        'valid': CustomDataset(args.data_path, args.data_type),
        'test': CustomDataset(args.data_path, args.data_type)
    }

    dataloader_dict = {
        'train':  DataLoader(dataset_dict['train'], 
                             collate_fn=PadCollate(), 
                             drop_last=True, 
                             pin_memory=True, 
                             batch_size=args.batch_size),
        'valid':  DataLoader(dataset_dict['valid'], 
                             collate_fn=PadCollate(), 
                             drop_last=True, 
                             pin_memory=True, 
                             batch_size=args.batch_size),
        'test':  DataLoader(dataset_dict['test'], 
                             collate_fn=PadCollate(), 
                             drop_last=True, 
                             pin_memory=True, 
                             batch_size=args.batch_size)
    }

    # Model setting
    #################################Fix##################################
    model = littleBert(vocab_num=32000, pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, 
                       max_len=args.max_len, d_model=args.d_model, d_embedding=args.d_embedding, 
                       n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                       n_layers=args.n_layers, dropout=args.dropout, device=device)
    model = model.to(device)

    # Optimizer Setting
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    lr_step_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma) # Decay LR by a factor of 0.1 every step_size

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        for phase in ['train', 'valid']:
            running_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Iterate over data
            


if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type', type=str, default='*.csv', help='data type')

    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate decay step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='learning rate decay rate')
