from ta.momentum import RSIIndicator #pip install ta 
from ta.trend import MACD, SMAIndicator
import pandas as pd
import pandas_ta as ta
import ta
import random
import numpy as np


class Features:
    def __init__(self, data):
        self.data = data
        pass

    def add_ema(self, source='Close'):
        self.data.df['EMA'] = ta.ema(self.data.df[source], length=200)

    def add_atr(self):
        self.data.df['ATR'] = ta.atr(self.data.df.High, self.data.df.Low, self.data.df.Close, length=7)

    def add_dummy(self):
        l = len(self.data.df)
        a = np.zeros(l)
        b = np.zeros(l)
        distance = 9
        for i in range(0, l-distance):
            r = random.uniform(0.3, 0.5 )
            v = self.data.df.Close.iloc[i+distance]
            a[i] = v * r
            b[i] = v * (1-r)
            a[i] = v
        self.data.df['A'] = a
        #self.data.df['B'] = b
        pass


    def add_rsi(self, source='Close'):
        rsi_indicator = ta.momentum.RSIIndicator(self.data.df[source], window=14)
        rsi = rsi_indicator.rsi()
        self.data.df['RSI'] = rsi

    def add_macd(self, source='Close'):
        macd_indicator = ta.trend.MACD(close=self.data.df[source], window_slow=26, window_fast=12, window_sign=9)
        self.data.df['MACD'] = macd_indicator.macd()
        self.data.df['MACD_Signal'] = macd_indicator.macd_signal()
        self.data.df['MACD_Diff'] = macd_indicator.macd_diff()

    def add_sma(self, source='Close'):
        self.data.df['SMA20'] = ta.trend.SMAIndicator(close=self.data.df[source], window=20).sma_indicator()
        self.data.df['SMA50'] = ta.trend.SMAIndicator(close=self.data.df[source], window=50).sma_indicator()
#        df['SMA100'] = SMAIndicator(close=df[source], window=100).sma_indicator()
#        df['SMA200'] = SMAIndicator(close=df[source], window=200).sma_indicator()

    def add_bollinger(self, source='Close'):
        self.data.df['BM'] = ta.volatility.BollingerBands(self.data.df[source]).bollinger_mavg()
        self.data.df['BU'] = ta.volatility.BollingerBands(self.data.df[source]).bollinger_hband()
        self.data.df['BL'] = ta.volatility.BollingerBands(self.data.df[source]).bollinger_lband()
