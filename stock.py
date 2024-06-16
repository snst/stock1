import yfinance as yf
import pandas as pd
import os

class Stock:
    def __init__(self):
        self.df = None
        self.name = None
        pass

    def make_cache_name(self, ticker):
        return f"data/{ticker}.csv"

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

        self.df = self.df[self.df['Volume'] != 0]

