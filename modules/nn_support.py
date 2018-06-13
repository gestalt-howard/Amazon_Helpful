# Script containing prerequisite functions for running a neural net
import pandas as pd
import numpy as np
import pickle
import os
import json
import shutil

import matplotlib.pyplot as plt


def save_history(filename, history):
    """Function for saving the epoch history JSON file"""
    with open(filename, mode='w', encoding='utf-8') as f:
        json.dump(history, f)


def load_history(filename):
    """Function for loading the epoch history JSON file"""
    with open(filename, mode='r', encoding='utf-8') as f:
        history = json.load(f)
    return(history)


def refresh_history(old, new):
    """Function for appending new data to JSON history file"""
    keys_list = ['val_loss', 'val_acc', 'loss', 'acc']
    for key in keys_list:
        new_entry = new[key]
        if len(new_entry) > 0:
            for elem in new_entry:
                old[key].append(elem)
        else:
            old[key].append(elem)
    return(old)


def create_folder_and_file(weights_folder, filename, history):
    """Function for creating folder and file"""
    print('Creating new training weights folder...')
    os.mkdir(weights_folder)
    print('Creating new JSON epoch history file...')
    save_history(filename, history)


def folder_gen(weights_folder, start_over=False):
    """Function for generating or removing training weights folder
    Also includes functionality to generate a JSON file for storing epoch history"""
    filename = weights_folder + 'history.json'
    history = {'val_loss': [], 'val_acc': [], 'loss': [], 'acc': []}
    if start_over:
        if os.path.exists(weights_folder):
            print('Deleting old training weights folder...')
            shutil.rmtree(weights_folder)
            create_folder_and_file(weights_folder, filename, history)
            return(filename)
        else:  # If folder doesn't exist...
            print('No training folder to remove')
            create_folder_and_file(weights_folder, filename, history)
            return(filename)
    # If start over isn't specified...
    if not os.path.exists(weights_folder):
        create_folder_and_file(weights_folder, filename, history)
        return(filename)
    else:
        print('Training weights folder already exists')
        return(filename)


def epoch_check(ext, template, weight_folder):
    """Function for checking the index of the latest epoch"""
    idx = -1
    filename = template + str(idx) + ext
    for file in os.listdir(weight_folder):
        if ext in file:
            num = int(file.split(template)[1].split(ext)[0])
            if num > idx:
                idx = num
                filename = str(file)
    return(idx, filename)


def visualize_epoch(history):
    """Function for visualizing epoch performance"""
    # Summarize history for accuracy
    plt.plot(history['acc'])  # Training accuracy
    plt.plot(history['val_acc'])  # Validation accuracy
    plt.title('Epoch Iteration vs. Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
