# Script to train random forest and make predictions
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from modules.get_ready import lets_train
from modules.get_ready import *

import os
import pandas as pd
import numpy as np
import pickle

# Load the data
debug = False
grid_search = False
train_folder, test_folder, train_df, train_labels, test_df = lets_train(debug)
reduced_train_idx, reduced_test_idx = get_reduced_training(debug)
# Trim and standardize
train_df, train_labels, test_df = trim_and_standardize(train_df, train_labels, test_df, reduced_train_idx, reduced_test_idx)


num_est = [10, 50, 100, 300, 600]
depth = [None, 3, 5, 7, 9]
max_feat = [3, 5, 7]
parameters = {'n_estimators': num_est, 'max_depth': depth, 'max_features': max_feat}
# Run Random Forest with grid search
if grid_search:
    rf_model = RandomForestRegressor()
    clf = GridSearchCV(rf_model, parameters, verbose=1, n_jobs=-1)
    print('Fitting the random forest regressor model using grid search...')
    clf.fit(train_df, train_labels)
    print('Best parameters:\n', clf.best_params_)
    # Get predictions
    predictions = clf.predict(test_df)
    train_pred = clf.predict(train_df)
# Plain random forest
else:
    rf_model = RandomForestRegressor(n_estimators=600, max_depth=10, verbose=1, n_jobs=-1)
    print('Fitting random forest regressor model...')
    rf_model.fit(train_df, train_labels)
    # Get predictions
    predictions = rf_model.predict(test_df)
    train_pred = rf_model.predict(train_df)

# Save test predictions
print('Saving test predictions into pickle file...')
with open(test_folder+'predictions_rf.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save train predictions
print('Saving train predictions into pickle file...')
with open(test_folder+'train_pred_rf.pickle', 'wb') as handle:
    pickle.dump(train_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
