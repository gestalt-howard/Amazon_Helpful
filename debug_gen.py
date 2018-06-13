# Script for generating debugging files for fast processing
from modules.unzip_files import *
import shutil
import pandas as pd
import pickle
import os

parent_path = os.getcwd() + '/'
debug_folder = 'debug_data/'
debug_path = parent_path + debug_folder

# Generate new debug folder
if os.path.exists(debug_path):
    print('Removing debug data folder...')
    shutil.rmtree(debug_path)
print('Generating folder for storing debugging data...')
os.mkdir(debug_path)

train_path = parent_path + 'train.json.gz'
test_path = parent_path + 'test_Helpful.json.gz'
# Load original datasets
print('Loading training and test zipped files...')
train_data = getDF(train_path)
test_data = getDF(test_path)

debug_train_name = debug_path + 'train.json'
debug_test_name = debug_path + 'test_Helpful.json'
# Save subset of data into newly generated folders
train_debug = train_data.sample(n=100)
test_debug = test_data.sample(n=100)
print('Saving training and testing debug JSON files...')
train_debug.to_json(debug_train_name)
test_debug.to_json(debug_test_name)
