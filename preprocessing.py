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
            time_dat_count = time_dat.groupby(time_dat['PULocationID']).count() # Counting
            new_dat = pd.DataFrame({
                'count':time_dat_count['VendorID'].tolist()
            })
            new_dat.index = time_dat_count.index.tolist() # because of pandas's weird setting
            if t == 0:
                total_dat = new_dat
            else:
                total_dat = pd.concat([total_dat, new_dat], axis=1) # Concat data

        total_dat.columns = list(set(dat1['pickup_time_index'])) # Column setting
        total_dat = total_dat.fillna(0) # Fill NaN by zero
        total_dat = total_dat.astype(int) # Type setting by integer

        month = i + 1
        total_dat.to_csv(f'./newyork_yellow_taxi_2019-0{month}_count.csv', index=False)

if __name__=='__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--data_type', type=str, default='*.csv', help='data type')
    args = parser.parse_args()
    main(args)
    print('Done!')