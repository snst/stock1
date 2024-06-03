import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Stock:
    def __init__(self):
        self.df = None
        self.name = None
        pass

    def make_cache_name(self, ticker):
        return f"{ticker}.csv"

    def load(self, ticker, start_date=None, end_date=None, interval="1d", reload=False):
        self.name = ticker
        file_path = self.make_cache_name(ticker)
        if reload or not os.path.exists(file_path):
            # Download data
            self.df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            # Cache data to a CSV file
            self.df.to_csv(file_path)
            print(f"Data ({len(self.df)}) cached to {file_path}")
        else:
            self.df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Data ({len(self.df)}) loaded from {file_path}")

    def drop_column(self, name):
        self.df = self.df.drop(name, axis=1)

    def head(self):
        print(self.df.head())

    def tail(self):
        print(self.df.tail())

    def info(self):
        print(self.df.iloc[0])
        print(self.df.iloc[-1])
        print(len(self.df))


    def trim(self, start, end):
        if start != None:
            self.df = self.df.iloc[start:]
        elif end != None:
            self.df = self.df.iloc[:end]
        else:
            self.df = self.df.iloc[start:end]

    def drop_nan(self):
        i = None
        for index, row in self.df.iterrows():
            if not row.isna().any():
                i = index
                break 
        if i != None:
            self.df = self.df[i:]

        for index, row in self.df.iterrows():
            if row.isna().any():
                print("data still contains NAN")
        pass


    def plot(self):
        #plt.ion()
        show_features=[
            #"Open", 
            "Close", 
            #"High", "Low", 
            "SMA20", "SMA50", "SMA100", "SMA200"]
        plt.figure(figsize=(20, 6))

        for f in show_features:
            plt.plot(self.df[f], label=f)

        # Add markers where column 'Y' has value 1
        markers_buy = self.df.index[self.df['Target'] == 1]
        markers_sell = self.df.index[self.df['Target'] == 0]
        plt.plot(markers_buy, self.df['Close'][markers_buy], 'go', label='buy')
        plt.plot(markers_sell, self.df['Close'][markers_sell], 'ro', label='sell')  # 'ro' means red circle
        plt.show()