# Script for finding "zero" indexes
from modules.get_ready import unzip_df

import os
import shutil
import pickle

# Get dataframes
debug = False
parent_path, [train, test] = unzip_df(debug)

# Set up reduced data folder
reduced_path = parent_path + 'reduced_data/'
if os.path.exists(reduced_path):
    print('Deleting old reduced data folder...')
    shutil.rmtree(reduced_path)
    print('Creating new reduced data folder...')
    os.mkdir(reduced_path)
else:
    print('Creating new reduced data folder...')
    os.mkdir(reduced_path)

# Find indexes of rows containing zero 'Out Of' values
train_reduced_idx = reduced_path + 'train_reduced_idx.pickle'
test_reduced_idx = reduced_path + 'test_reduced_idx.pickle'
file_paths = [train_reduced_idx, test_reduced_idx]
for ind, df in enumerate([train, test]):
    zeros_list = []
    help_ratings = df.helpful
    help_idx = list(help_ratings.index)
    help_dict = help_ratings.values
    for i, dict in enumerate(help_dict):
        if dict['outOf']!=0:
            zeros_list.append(i)
    print('Saving', file_paths[ind])
    with open(file_paths[ind], 'wb') as handle:
        pickle.dump(zeros_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
