# Script for running multilayer perceptron
from modules.get_ready import *
from modules.nn_support import *
from modules.neural_models import *

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import pandas as pd
import pickle
import pdb
import os


# Neural net parameters
train_flag = False
pred_flag = True
max_epochs = 100

# Get data
debug = False
train_folder, test_folder, train_df, train_labels, test_df = lets_train(debug)
reduced_train_idx, reduced_test_idx = get_reduced_training(debug)

# Trim and standardize
train_df, train_labels, test_df = trim_and_standardize(train_df, train_labels, test_df, reduced_train_idx, reduced_test_idx)

# Make train and validation splits
x_train, x_val, y_train, y_val = train_test_split(train_df, train_labels, test_size=0.2, random_state=0)
model_size = (len(x_train[0]), )

# Create folder for storing epoch weights
start_over = False
weights_folder = train_folder + 'multi_weights/'
history_name = folder_gen(weights_folder, start_over)

# Check for any existing weight files:
weight_template = 'multi_'
weight_ext = '.h5'
latest_idx, latest_file = epoch_check(weight_ext, weight_template, weights_folder)
# Define callback automatic save file name template
filepath = weights_folder + weight_template + '{epoch:02d}' + weight_ext

# Train model or load weights
model = multilayer(model_size)  # Initialize multilayer perceptron model
if train_flag:
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]
    print('Generating multilayer perceptron weights...')
    if latest_idx != -1:
        model.load_weights(weights_folder + latest_file)
        new_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, initial_epoch=latest_idx, epochs=max_epochs, callbacks=callbacks_list)
    else:
        new_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, epochs=max_epochs, callbacks=callbacks_list)
    # Append new epoch history to old history file
    old_history = load_history(history_name)
    updated_history = refresh_history(old_history, new_history.history)
    save_history(history_name, updated_history)

    # Visualize epoch performance
    visualize_epoch(updated_history)

# Get predictions
if pred_flag:
    # Choose best epoch
    latest_idx, latest_file = epoch_check(weight_ext, weight_template, weights_folder)
    best_val = str(input('Based on the accuracy plot, choose the best epoch (1-%s)'%latest_idx))
    if len(best_val) < 2:
        best_val = '0' + best_val
    filename = weights_folder + weight_template + best_val + weight_ext

    if latest_idx == -1:
        print('No weights have been generated yet...')
    else:  # Execute predictions
        print('Loading weights file...')
        model.load_weights(filename)
        # Test predictions
        predictions = model.predict(test_df, verbose=1)
        flat_pred = np.asarray([x[0] for x in predictions])
        # Train predictions
        train_pred = model.predict(train_df, verbose=1)
        flat_train_pred = np.asarray([x[0] for x in train_pred])

        # Save predictions
        print('Saving test predictions into pickle file...')
        with open(test_folder+'predictions_multi.pickle', 'wb') as handle:
            pickle.dump(flat_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving train predictions into pickle file...')
        with open(test_folder+'train_pred_multi.pickle', 'wb') as handle:
            pickle.dump(flat_train_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
