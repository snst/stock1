import yfinance as yf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

def p(a, b):
    return (100 / a * b) - 100

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

        self.df = self.df[self.df['Volume'] != 0]


    def preprocess(self):
        self.drop_column('Adj Close')
        self.drop_column('Volume')
        #self.df.reset_index(drop=True, inplace=True)
        self.df.isna().sum()


    def add_target(self, distance=5, percent=1):
        BUY = 1
        SELL = 2
        #self.df['Target'] = 0
        l = len(self.df)
        target = np.zeros(l)
        for i in range(0, l-distance):
            I = self.df.Close.iloc[i]
            for k in range(0, distance):
                K = self.df.Close.iloc[i+k]
                if p(I, K) < -0.1:
                    break
                if p(I, K) > percent:
                    target[i] = BUY
                    break
            for k in range(0, distance):
                K = self.df.Close.iloc[i+k]
                if p(I, K) < -percent:
                    target[i] = SELL
                    break
        self.df['Target'] = target
        pass



    def pivotid(self, df1, time_index, n1, n2): #n1 n2 before and after candle l
        #https://colab.research.google.com/drive/1ATNIwG-gYUHs3BfHrsyfeS7ctTxBGW1t#scrollTo=964219a4
        #https://www.youtube.com/watch?v=MkecdbFPmFY
        l = df1.index.get_loc(time_index)
        if l-n1 < 0 or l+n2 >= len(df1):
            return 0
        
        pividlow=1
        pividhigh=1
        first_i = l-n1
        last_i = l+n2
        for i in range(l-n1, l+n2+1):
#            if(df1.Low.iloc[l]>df1.Low.iloc[i]):
            if(df1.Close.iloc[l]>df1.Close.iloc[i]):
                pividlow=0
#            if(df1.High.iloc[l]<df1.High.iloc[i]):
            if(df1.Close.iloc[l]<df1.Close.iloc[i]):
                pividhigh=0
            """
            if i == first_i and pividlow:
                percent = (100 / df1.Close.iloc[l] * df1.Close.iloc[i]) - 100
                if percent < 1:
                    pividlow=0
                pass
            if i == last_i and pividhigh:
                percent = (100 / df1.Close.iloc[l] * df1.Close.iloc[i]) - 100
                if percent < -1:
                    pividhigh=0
                pass
            """
        if pividlow and pividhigh:
            return 3
        elif pividlow:
            return 1
        elif pividhigh:
            return 2
        else:
            return 0
        
    def apply_pivot(self, dist=10):
        self.df['Pivot'] = self.df.apply(lambda x: self.pivotid(self.df, x.name, dist, dist), axis=1)

    def pointpos(self, x):
        if x['Pivot']==1:
            return x['Low']-1e-3
        elif x['Pivot']==2:
            return x['High']+1e-3
        else:
            return np.nan

    def apply_pointpos(self):
        self.df['Pointpos'] = self.df.apply(lambda row: self.pointpos(row), axis=1)


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



def load_stocks(stock_names, start_date, end_date, reload, features, sequencer, target):
    s = None
    for stock_name in stock_names:
        s = Stock()
        s.load(stock_name, start_date=start_date, end_date=end_date, reload=reload)
        s.drop_column('Adj Close')

        features.add_all(s.df)
        s.drop_nan()
        features.save_columns(s.df)
        target.add_target(s.df)
        features.save_columns(s.df)
        #rows_with_nan = s.df.isna().any(axis=1)
        pass

        #s.info()
        #s.plot()

        sequencer.add(s.df)
    return s