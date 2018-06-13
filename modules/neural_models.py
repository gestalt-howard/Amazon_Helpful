# Script containing neural network models
import tensorflow as tf
import numpy as np
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD


# Fully connected network (multilayer perceptron)
def multilayer(size):
    inputs = Input(size)
    # 1st hidden layer
    dense1 = Dense(2048, name='dense1')(inputs)
    dense2 = Dropout(0.2)(dense1)
    # 2nd hidden layer
    dense3 = Dense(512, name='dense3')(dense2)
    dense4 = Dropout(0.2)(dense3)
    # 3rd hidden layer
    dense5 = Dense(128, name='dense5')(dense4)
    dense6 = Dropout(0.2)(dense5)
    # Final layer
    densef = Dense(1, activation='linear', name='densef')(dense6)

    model = Model(inputs=inputs, outputs=densef)
    # Optimizer
    opt = SGD(lr=0.00001)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])
    return(model)
