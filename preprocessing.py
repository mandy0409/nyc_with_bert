# Import Modules
import os
import time
import argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

def main(args):

    # Data list setting
    print('Data Loading...')
    start_time = time.time()

    data_path = './data/'
    data_list = sorted(glob(os.path.join(args.data_path, args.data_type)))
    print(f'It will {len(data_list)} time loop...')

    for i, data in enumerate(data_list):
        # Read Data
        dat1 = pd.read_csv(data)

        # Count by pickup time index
        for t in tqdm(range(48)):
            time_dat = dat1[dat1['pickup_time_index'] == t] # Time data setting
            groupby_index = [time_dat['PULocationID'], time_dat['pickup_weekday_index']]
            time_dat_count = time_dat.groupby(groupby_index).count() # Counting
            index_list = time_dat_count.index.tolist()
            
            # Make dataframe
            new_dat = pd.DataFrame({
                'location': [x[0] for x in index_list],
                'weekday': [x[1] for x in index_list],
                'hour': [t for _ in range(len(index_list))],
                'count': time_dat_count['VendorID'].tolist()
            })
            
            # Concatenate
            if t == 0:
                concat_dat = new_dat
            else:
                concat_dat = pd.concat([concat_dat, new_dat], axis=0) # Concat data

        # Prepare to merge data
        total_list = list()
        for l in set(concat_dat['location']):
            for w in set(concat_dat['weekday']):
                for h in set(concat_dat['hour']):
                    total_list.append(str(l) + '/' + str(w) + '/' + str(h))

        all_dat = pd.DataFrame([x.split('/') for x in total_list])

        # all_dat preprocessing
        column_name = ['location', 'weekday', 'hour']
        all_dat.columns = column_name
        all_dat = all_dat.astype('int')

        # Merge data
        total_dat = pd.merge(left=all_dat, right=concat_dat, on=column_name, how='left')

        # Save to csv file
        month = i + 1
        total_dat.to_csv(os.path.join(args.data_path, f'newyork_yellow_taxi_2019-0{month}_count.csv'), index=False)

if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type', type=str, default='*.csv', help='data type')
    args = parser.parse_args()
    main(args)
    print('Done!')