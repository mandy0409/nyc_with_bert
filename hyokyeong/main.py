# Import Module
import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

# Import Custom
from model.bert import littleBert
from dataset import CustomDataset, getDataLoader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Data Loading...')
    start_time = time.time()

    data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))

    dataset_dict = {
        'train': CustomDataset(data_list[0]),
        'valid': CustomDataset(data_list[1])
    }

    dataloader_dict = {
        'train':  getDataLoader(dataset_dict['train'], 
                                shuffle=True,
                                drop_last=True, 
                                pin_memory=True, 
                                batch_size=args.batch_size,
                                num_workers=args.num_workers),
        'valid':  getDataLoader(dataset_dict['valid'], 
                                shuffle=True, 
                                drop_last=True, 
                                pin_memory=True, 
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
    }

    # Model setting
    model = littleBert(pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, 
                       max_len=args.max_len, d_model=args.d_model, d_embedding=args.d_embedding, 
                       n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                       n_layers=args.n_layers, dropout=args.dropout, device=device)
    model = model.to(device)

    # Optimizer Setting
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    lr_step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma) # Decay LR by a factor of 0.1 every step_size

    # Preparing
    best_val_loss = None
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H_%M_%S')
    if not os.path.exists('./save/'):
        os.mkdir('./save/')
    if not os.path.exists(f'./save/save_{nowDatetime}'):
        os.mkdir(f'./save/save_{nowDatetime}')
    hyper_parameter_setting = dict()
    hyper_parameter_setting['n_layers'] = args.n_layers
    hyper_parameter_setting['d_model'] = args.d_model
    hyper_parameter_setting['n_head'] = args.n_head
    hyper_parameter_setting['d_embedding'] = args.d_embedding
    hyper_parameter_setting['dim_feedforward'] = args.dim_feedforward
    with open(f'./save/save_{nowDatetime}/hyper_parameter_setting.txt', 'w') as f:
        for key in hyper_parameter_setting.keys():
            f.write(str(key) + ': ' + str(hyper_parameter_setting[key]))
            f.write('\n')

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Setting done...! / {spend_time}min spend...!')

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            running_loss = 0
            freq = args.print_freq - 1
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Iterate over data
            for i, input_ in enumerate(tqdm(dataloader_dict[phase])):

                # Input to Device(CUDA) and split
                src = input_[0].to(device)
                src_hour = input_[2].to(device)
                src_weekday = input_[3].to(device)
                trg = input_[4].to(device)

                # Optimizer Setting
                optimizer.zero_grad()

                # Model Training & Validation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(src, src_hour, src_weekday)

                    # Backpropagate Loss
                    loss = criterion(outputs, trg.to(torch.float))
                    if phase == 'train':
                        loss.backward()
                        torch_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                        # Print every print_frequency
                        freq += 1
                        if freq == args.print_freq:
                            total_loss = loss.item()
                            print("[loss:%5.2f]" % (total_loss))
                            total_loss_list.append(total_loss)
                            freq = 0
                    if phase == 'valid':
                        val_loss += loss.item()

            # Save model and view total loss
            if phase == 'valid': 
                print('='*45)
                val_loss /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | spend_time:%5.2fmin"
                        % (e, val_loss, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    val_loss_save = round(val_loss, 2)
                    torch.save(model.state_dict(), f'./save/save_{nowDatetime}/model_{e}_{val_loss_save}.pt')
                    best_val_loss = val_loss

        # Gradient Scheduler Step
        scheduler.step()

if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type', type=str, default='*.h5', help='data type')
    parser.add_argument('--print_freq', type=int, default=100, help='loss print frequency')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of cpu works')

    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--pad_idx', type=int, default=0, help='padding index')
    parser.add_argument('--bos_idx', type=int, default=1, help='start token index')
    parser.add_argument('--eos_idx', type=int, default=2, help='end token index')
    parser.add_argument('--max_len', type=int, default=300, help='max length of input')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
    parser.add_argument('--d_embedding', type=int, default=256, help='dimension of embedding')
    parser.add_argument('--n_head', type=int, default=8, help='''multihead-attention's head count''')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='layers number')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate decay step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--grad_clip', type=int, default=5, help='gradient clipping norm')
    args = parser.parse_args()

    main(args)
    # data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))

    # # print(glob(os.path.join(args.data_path, args.data_type)))
    # print(data_list[0])
    # print(CustomDataset(data_list[0]))
    print('Done!')