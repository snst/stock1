import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from stock import *

class Sequencer:
    def __init__(self, len):
        self.x = None
        self.y = None
        self.sequence_len = len
        pass

    def add(self, df):
        xa = []
        ya = []
        for i in range(len(df) - self.sequence_len):
            xa.append(df.iloc[i:i+self.sequence_len, :-1].values)
            ya.append(df.iloc[i+self.sequence_len, -1])  # Target value
            x = np.array(xa)
            y = np.array(ya)
        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = np.concatenate((self.x, x), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)
        pass


    def split_train_test(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size)