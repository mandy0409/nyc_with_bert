# Import Modules
import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

def main(args):

    # Data list setting
    print('Data Loading...')

    data_path = './data/'
    data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))
    print(f'It will {len(data_list)} time loop...')

    # List and DataFrame setting
    input_train_list = list()
    output_train_list = list()
    weekday_train_list = list()
    hour_train_list = list()

    input_valid_list = list()
    output_valid_list = list()
    weekday_valid_list = list()
    hour_valid_list = list()

    train_dat = pd.DataFrame()
    valid_dat = pd.DataFrame()

    for i, data in enumerate(data_list):
        # Read Data
        dat1 = pd.read_csv(data)

        # Month setting
        month = i + 1

        # Count by pickup time index
        for t in tqdm(range(48)):
            time_dat = dat1[dat1['pickup_time_index'] == t] # Time data setting
            groupby_index = [time_dat['PULocationID'], time_dat['pickup_weekday_index']]
            time_dat_count = time_dat.groupby(groupby_index).count() # Counting
            index_list = time_dat_count.index.tolist()
            
            # Make dataframe
            new_dat = pd.DataFrame({
                'month': [month for _ in range(len(index_list))],
                'location': [x[0] for x in index_list],
                'weekday': [x[1] for x in index_list],
                'hour': [t for _ in range(len(index_list))],
                'count': time_dat_count['VendorID'].tolist()
            })
            
            # Concatenate
            if t == 0:
                total_dat = new_dat
            else:
                total_dat = pd.concat([total_dat, new_dat], axis=0) # Concat data

        total_dat = total_dat.fillna(0)
        total_dat.to_csv(os.path.join(args.data_path, f'newyork_yellow_taxi_2019-0{month}_count.csv'), index=False)

        if month <= args.data_split_month:
            train_dat = pd.concat([train_dat, total_dat], axis=0)
        else:
            valid_dat = pd.concat([valid_dat, total_dat], axis=0)

    print('Data setting done!')
    
    print('Train data saving...')
    for l in tqdm(set(train_dat['location'])):
        location_dat = train_dat[train_dat['location'] == l]
        location_dat = location_dat.sort_values(by=['month','weekday','hour'])
        for t in range(len(location_dat) - args.input_timestep - args.output_timestep):
            input_ = location_dat.iloc[t:t+args.input_timestep]['count'].tolist()
            weekday_ = location_dat.iloc[t:t+args.input_timestep]['weekday'].tolist()
            hour_ = location_dat.iloc[t:t+args.input_timestep]['hour'].tolist()
            output_ = location_dat.iloc[t+args.input_timestep:t+args.input_timestep+args.output_timestep]['count'].tolist()
            input_train_list.append(input_)
            weekday_train_list.append(weekday_)
            hour_train_list.append(hour_)
            output_train_list.append(output_)

    hf_data_train = h5py.File(os.path.join(args.data_path, 'preprocessed_train.h5'), 'w')
    hf_data_train.create_dataset(f'train_src', data=input_train_list)
    hf_data_train.create_dataset(f'train_src_week', data=weekday_train_list)
    hf_data_train.create_dataset(f'train_src_hour', data=hour_train_list)
    hf_data_train.create_dataset(f'train_trg', data=output_train_list)
    hf_data_train.close()

    print('Valid data saving...')
    for l in tqdm(set(valid_dat['location'])):
        location_dat = valid_dat[valid_dat['location'] == l]
        location_dat = location_dat.sort_values(by=['month','weekday','hour'])
        for t in range(len(location_dat) - args.input_timestep - args.output_timestep):
            input_ = location_dat.iloc[t:t+args.input_timestep]['count'].tolist()
            weekday_ = location_dat.iloc[t:t+args.input_timestep]['weekday'].tolist()
            hour_ = location_dat.iloc[t:t+args.input_timestep]['hour'].tolist()
            output_ = location_dat.iloc[t+args.input_timestep:t+args.input_timestep+args.output_timestep]['count'].tolist()
            input_valid_list.append(input_)
            weekday_valid_list.append(weekday_)
            hour_valid_list.append(hour_)
            output_valid_list.append(output_)

    hf_data_valid = h5py.File(os.path.join(args.data_path, 'preprocessed_valid.h5'), 'w')
    hf_data_valid.create_dataset(f'valid_src', data=input_valid_list)
    hf_data_valid.create_dataset(f'valid_src_week', data=weekday_valid_list)
    hf_data_valid.create_dataset(f'valid_src_hour', data=hour_valid_list)
    hf_data_valid.create_dataset(f'valid_trg', data=output_valid_list)
    hf_data_valid.close()

if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type', type=str, default='*.csv', help='data type')
    parser.add_argument('--data_split_month', type=int, default=3, help='baseline of data split month')
    parser.add_argument('--input_timestep', type=int, default=12, help='input time step')
    parser.add_argument('--output_timestep', type=int, default=12, help='output time step')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print(f'{round((time.time()-start_time)/60, 4)}min spend...')
    print('Done!')