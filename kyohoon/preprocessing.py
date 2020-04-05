# Import Modules
import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from random import shuffle
from datetime import datetime
from itertools import product

def main(args):

    # Data list setting
    print('Data Loading...')

    data_path = './data/'
    data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))
    print(f'It will {len(data_list)} time loop...')

    total_count_dat = pd.DataFrame()

    for i, data in enumerate(data_list):
        # Read Data
        dat1 = pd.read_csv(data)
        
        # Month setting
        month = i + 1
        print(f'{month} month start...')

        # Pre-process other month
        if month <= 9:
            year_month = f'2019-0{month}'
        else:
            year_month = f'2019-{month}'
            
        # Unique list setting
        date_list = [x for x in sorted(list(set(dat1['pickup_date']))) if x[:7] == year_month]
        hour_list = range(0,48)
        location_list = list(set(dat1['PULocationID']))

        # Make processed list
        location_list2, date_list2, hour_list2, weekday_list = list(), list(), list(), list()

        for location, date, hour in product(location_list, date_list, hour_list):
            location_list2.append(location)
            date_list2.append(date)
            hour_list2.append(hour)
            weekday_list.append(datetime.strptime(date, '%Y-%m-%d').weekday())

        # Count
        count_list = list()

        for i in tqdm(range(len(location_list2))):
            location_dat = dat1[dat1['PULocationID'] == location_list2[i]]
            date_dat = location_dat[location_dat['pickup_date'] == date_list2[i]]
            hour_dat = date_dat[date_dat['pickup_time_index'] == hour_list2[i]]
            count_list.append(len(hour_dat))
    
        # Total_data make & save
        total_dat = pd.DataFrame({
            'location': location_list2,
            'date': date_list2,
            'weekday': weekday_list,
            'hour': hour_list2,
            'count': count_list
        })
        total_dat.to_csv(os.path.join(args.data_path, f'newyork_yellow_taxi_2019-0{month}_count.csv'), index=False)

        # Concat
        total_count_dat = pd.concat([total_count_dat, total_dat])
    
    total_count_dat.to_csv(os.path.join(args.data_path, 'newyork_yellow_taxi_total_count.csv'))

    print('Count data save done!')
    print('H5 processing & train, valid split...')

    # Total data preprocessing
    total_count_dat = total_count_dat.sort_values(['location', 'date', 'hour'])
    input_, weekday_, hour_, output_ = list(), list(), list(), list()

    for l in tqdm(set(total_dat['location'])):
        l_dat = total_dat[total_dat['location'] == l]
        for i in range(len(l_dat) - 24):
            src_dat = l_dat[i:i+12]
            input_.append(src_dat['count'].tolist())
            weekday_.append(src_dat['weekday'].tolist())
            hour_.append(src_dat['hour'].tolist())
            trg_dat = l_dat[i+12:i+24]
            output_.append(trg_dat['count'].tolist())

    # Train & Validation set split
    ix = list(range(len(input_)))
    shuffle(ix)

    train_ix = ix[:int(len(ix) * 0.8)]
    valid_ix = ix[int(len(ix) * 0.8):]

    # H5 save
    print('Saving...')
    hf_data_train = h5py.File(os.path.join(args.data_path, 'preprocessed_train.h5'), 'w')
    hf_data_train.create_dataset(f'train_src', data=[input_[i] for i in train_ix])
    hf_data_train.create_dataset(f'train_src_week', data=[weekday_[i] for i in train_ix])
    hf_data_train.create_dataset(f'train_src_hour', data=[hour_[i] for i in train_ix])
    hf_data_train.create_dataset(f'train_trg', data=[output_[i] for i in train_ix])
    hf_data_train.close()

    hf_data_valid = h5py.File(os.path.join(args.data_path, 'preprocessed_valid.h5'), 'w')
    hf_data_valid.create_dataset(f'valid_src', data=[input_[i] for i in valid_ix])
    hf_data_valid.create_dataset(f'valid_src_week', data=[weekday_[i] for i in valid_ix])
    hf_data_valid.create_dataset(f'valid_src_hour', data=[hour_[i] for i in valid_ix])
    hf_data_valid.create_dataset(f'valid_trg', data=[output_[i] for i in valid_ix])
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