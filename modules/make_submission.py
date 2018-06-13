# This script contains a function for creating the Kaggle submission file
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pdb
import numpy as np

def submission(format, predict, idx, save):
    format_df = pd.read_csv(format)
    with open(predict, 'rb') as handle:
        predict_vals = pickle.load(handle)
    with open(idx, 'rb') as handle:
        t_idx = pickle.load(handle)
    # Fill in the prediction values
    for num, i in enumerate(t_idx):
        format_df.prediction.iloc[i] = predict_vals[num]
    # Create submission dataframe
    submit_df = format_df[format_df.prediction.notnull()]

    # Save submission file
    submit_df.to_csv(save, index=None)
