# Data preprocessing script
from modules.preprocessing import preprocessing
from modules.get_ready import *
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import pdb


# Load corpus vocabulary, Word2Vec trained model, and categories
corpus_folder, train_vocab = load_corpus()
print('Loading Word2Vec trained model...')
model = Word2Vec.load(corpus_folder + 'word_model.bin')
print('Loading categories dictionary...')
with open(corpus_folder + 'categories.pickle', 'rb') as handle:
    categs_dict = pickle.load(handle)

# Load dataframes and save sampled indexes
debug = False
parent_path, df_build = unzip_df(debug)
test_idx = [idx for idx, row in df_build[1].iterrows()]

# Generate data files
data_types = ['train', 'test']
for idx, dtype in enumerate(data_types):
    data_dir = parent_path + '%s_data/'%dtype
    raw_data_name = '%s_raw_data.pickle'%dtype
    pro_data_name = '%s_processed_data.csv'%dtype
    score_name = '%s_true_scores.pickle'%dtype

    struct = preprocessing(df_build[idx], train_vocab, categs_dict)
    # Generate data if it doesn't exist
    if os.path.exists(data_dir):
        print('Removing old data folder for %s data'%dtype)
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        print('Creating new data folder for %s data'%dtype)
        if dtype=='train':
            struct.true_scores(data_dir, score_name)
        # Generate raw and processed data
        struct.process_data(data_dir, raw_data_name, model)
        struct.assemble_dataset(data_dir, raw_data_name, pro_data_name)

# Save test indexes
test_idx_fname = parent_path + '/test_data/' + 'test_idx.pickle'
print('Saving test indexes...')
with open(test_idx_fname, 'wb') as handle:
    pickle.dump(test_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
