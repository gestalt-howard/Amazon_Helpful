# Script for setting up directory structure
from sklearn.preprocessing import StandardScaler
from modules.unzip_files import *

import os
import pdb
import pandas as pd
import pickle
import collections


def load_corpus():
    """Function for loading training corpus vocabulary"""
    # Define directory structure
    parent_path =  os.getcwd() + '/'
    corpus_path = parent_path + 'corpus_data/'
    corpus_name = corpus_path + 'train_corpus_vocab.pickle'
    # Load corpus vocabulary
    with open(corpus_name, 'rb') as handle:
        train_vocab = pickle.load(handle)
    return(corpus_path, train_vocab)


def setup_debug(debug_flag):
    """Function for changing parent path"""
    parent_path =  os.getcwd() + '/'
    if debug_flag:
        return(parent_path + 'debug_data/')
    else:
        return(parent_path)


def unzip_df(debug_flag):
    """Function for setting up directory structure
    Input: debug_flag (True or False)
    Returns: new parent path and list of dataframe objects (train, test)"""
    # Initial file / folder name designations
    train_zip = 'train.json.gz'
    test_zip = 'test_Helpful.json.gz'

    if debug_flag:
        print('Loading debugging data...')
        parent_path = setup_debug(debug_flag)
        train_zip = train_zip.split('.')[0] + '.json'
        test_zip = test_zip.split('.')[0] + '.json'
        df_build = [pd.read_json(parent_path + train_zip),
                    pd.read_json(parent_path + test_zip)]
    else:
        print('Loading real training and test data...')
        parent_path = setup_debug(debug_flag)
        df_build = [getDF(parent_path + train_zip), getDF(parent_path + test_zip)]

    return(parent_path, df_build)


def lets_train(debug_flag):
    """Function for loading processed datasets and setting directory structure"""
    # Directory structure
    parent_path = setup_debug(debug_flag)
    train_folder = parent_path + 'train_data/'
    test_folder = parent_path + 'test_data/'
    # File names
    train_label_path = train_folder + 'train_true_scores.pickle'
    train_data_path = train_folder + 'train_processed_data.csv'
    test_data_path = test_folder + 'test_processed_data.csv'

    # Training data and labels
    print('Loading training data...')
    train_df = pd.read_csv(train_data_path, index_col=0)
    with open(train_label_path, 'rb') as handle:
        train_labels = pickle.load(handle)
    # Test data
    print('Loading test data...')
    test_df = pd.read_csv(test_data_path, index_col=0)

    return(train_folder, test_folder, train_df, train_labels, test_df)


def flatten(l):
    """Recursively returns items from a list"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def text_processor(sentence, table):
    """Function for removing punctuations and uncasing"""
    sentence = sentence.translate(table)
    return([w.lower() for w in sentence.split() if w!=' ' and w!=None])


def get_reduced_training(debug):
    """Function for loading the indexes of zero prediction rows"""
    parent_path = setup_debug(debug)
    reduced_folder = parent_path + 'reduced_data/'
    reduced_train = reduced_folder + 'train_reduced_idx.pickle'
    reduced_test = reduced_folder + 'test_reduced_idx.pickle'
    print('Loading indexes for reduced data...')
    with open(reduced_train, 'rb') as handle:
        reduced_train_idx = pickle.load(handle)
    with open(reduced_test, 'rb') as handle:
        reduced_test_idx = pickle.load(handle)
    return(reduced_train_idx, reduced_test_idx)


def perform_reduce(data, idx_list):
    """Function for eliminating the zero prediction rows"""
    bool_mask = np.zeros(len(data), dtype=np.bool)
    bool_mask[idx_list] = True
    data = data[bool_mask]
    return(data)


def trim_and_standardize(train_df, train_labels, test_df, train_idx, test_idx):
    """Function for trimming and standardizing inputs"""
    # Trim
    train_reduced = perform_reduce(train_df.as_matrix(), train_idx)
    labels_reduced = perform_reduce(np.asarray(train_labels), train_idx)
    test_reduced = perform_reduce(test_df.as_matrix(), test_idx)
    # Standardize
    scaler = StandardScaler()
    train = scaler.fit_transform(train_reduced)
    test = scaler.fit_transform(test_reduced)
    return(train, labels_reduced, test)
