import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from stock import *


class Sequencer:
    def __init__(self, len):
        self.X = None
        self.y = None
        self.y_cat = None
        self.d = None
        self.sequence_len = len
        pass

    def add(self, df):
        xa = []
        ya = []
        da = []
        for i in range(len(df) - self.sequence_len):
            xa.append(df.iloc[i:i+self.sequence_len, :-1].values)
            ya.append(df.iloc[i+self.sequence_len-1, -1])  # Target value
            da.append(df.index[i+self.sequence_len-1])  # Target value
            X = np.array(xa)
            y = np.array(ya)
            d = np.array(da)
        if self.X is None:
            self.X = X
            self.y = y
            self.d = d
        else:
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)
            self.d = np.concatenate((self.d, d), axis=0)
        self.y_cat = to_categorical(self.y, num_classes=3)
        pass

    def split_train_test(self, test_size=0.2, shuffle=False):
        #self.X2 = self.X.reshape((self.X.shape[0], self.X.shape[1], 1, 3))
        #X = X.reshape((X.shape[0], X.shape[1], 1, num_features))
        #self.X = self.X2
        y = to_categorical(self.y, num_classes=3)
        self.X_train, self.X_test, self.y_train, self.y_test, self.d_train, self.d_test = train_test_split(
            self.X, y, self.d, test_size=test_size, shuffle=shuffle)
        pass

    def get_data_train(self):
        return self.X_train, self.y_train, self.d_train

    def get_data_test(self):
        return self.X_test, self.y_test, self.d_test

    def get_data_all(self):
        return self.X, self.y, self.d                