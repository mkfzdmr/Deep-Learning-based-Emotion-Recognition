# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Activation, Layer
import argparse
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import backend as K

from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Dropout
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import datetime


class networkArchFonc:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(filters=32,
                         kernel_size=5,
                         padding="same",
                         activation="relu",
                         input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=64, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=128, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.summary()
        model.add(Reshape((128, 3), name='predictions'))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        # return the constructed network architecture
        return model
