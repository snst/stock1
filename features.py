from ta.momentum import RSIIndicator #pip install ta 
from ta.trend import MACD, SMAIndicator
import pandas as pd


class Features:
    def __init__(self):
        self.column_dict = {}
        pass

    def add_rsi(self, df, source='Close'):
        rsi_indicator = RSIIndicator(df[source], window=14)
        rsi = rsi_indicator.rsi()
        df['RSI'] = rsi

    def add_macd(self, df, source='Close'):
        macd_indicator = MACD(close=df[source], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Diff'] = macd_indicator.macd_diff()

    def add_sma(self, df, source='Close'):
        df['SMA20'] = SMAIndicator(close=df[source], window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(close=df[source], window=50).sma_indicator()
#        df['SMA100'] = SMAIndicator(close=df[source], window=100).sma_indicator()
#        df['SMA200'] = SMAIndicator(close=df[source], window=200).sma_indicator()

    def add_all(self, df):
        self.add_rsi(df)
        self.add_macd(df)
        self.add_sma(df)    

    def save_columns(self, df):
        for name in df.columns:
            index = df.columns.get_loc(name)
            if not name in self.column_dict:
                self.column_dict[name] = index
            else:
                if index != self.column_dict[name]:
                    print("INVALID COLUMN INDEX")