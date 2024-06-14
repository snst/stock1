from ta.momentum import RSIIndicator #pip install ta 
from ta.trend import MACD, SMAIndicator
import pandas as pd
import pandas_ta as ta
import ta


class Features:
    def __init__(self, df):
        self.column_dict = {}
        self.df = df
        pass

    def add_ema(self, source='Close'):
        self.df['EMA'] = ta.ema(self.df[source], length=200)

    def add_atr(self):
        self.df['ATR'] = ta.atr(self.df.High, self.df.Low, self.df.Close, length=7)

    def add_rsi(self, source='Close'):
        rsi_indicator = ta.momentum.RSIIndicator(self.df[source], window=14)
        rsi = rsi_indicator.rsi()
        self.df['RSI'] = rsi

    def add_macd(self, source='Close'):
        macd_indicator = ta.trend.MACD(close=self.df[source], window_slow=26, window_fast=12, window_sign=9)
        self.df['MACD'] = macd_indicator.macd()
        self.df['MACD_Signal'] = macd_indicator.macd_signal()
        self.df['MACD_Diff'] = macd_indicator.macd_diff()

    def add_sma(self, source='Close'):
        self.df['SMA20'] = ta.trend.SMAIndicator(close=self.df[source], window=20).sma_indicator()
        self.df['SMA50'] = ta.trend.SMAIndicator(close=self.df[source], window=50).sma_indicator()
#        df['SMA100'] = SMAIndicator(close=df[source], window=100).sma_indicator()
#        df['SMA200'] = SMAIndicator(close=df[source], window=200).sma_indicator()

    def add_bollinger(self, source='Close'):
        self.df['BM'] = ta.volatility.BollingerBands(self.df[source]).bollinger_mavg()
        self.df['BU'] = ta.volatility.BollingerBands(self.df[source]).bollinger_hband()
        self.df['BL'] = ta.volatility.BollingerBands(self.df[source]).bollinger_lband()

    def save_columns(self):
        for name in self.df.columns:
            index = self.df.columns.get_loc(name)
            if not name in self.column_dict:
                self.column_dict[name] = index
            else:
                if index != self.column_dict[name]:
                    print("INVALID COLUMN INDEX")