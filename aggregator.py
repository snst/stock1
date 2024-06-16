import numpy as np
import pandas as pd
from myutils import *


class Aggregator:
    def __init__(self, stock):
        self.df = stock.df
        self.name = stock.name
        self.column_dict = {}

    def drop_not_needed_stock_columns(self):
        self.drop_column('Adj Close')
        self.drop_column('Volume')
        self.drop_column('High')
        self.drop_column('Low')
        self.drop_column('Open')
        #self.df.reset_index(drop=True, inplace=True)
        #self.df.isna().sum()

    def drop_not_needed_fear_and_greed_columns(self):
        #self.drop_column('fear_and_greed_historical_val')
        self.drop_column('market_momentum_sp500_val')
        self.drop_column('market_momentum_sp125_val')
        #self.drop_column('stock_price_strength_val')
        #self.drop_column('stock_price_breadth_val')
        #self.drop_column('put_call_options_val')
        #self.drop_column('market_volatility_vix_val')
        #self.drop_column('market_volatility_vix_50_val')
        #self.drop_column('junk_bond_demand_val')
        #self.drop_column('safe_haven_demand_val')
        for col in self.df.columns:
            if col.endswith('_rating'):
                self.drop_column(col)

    def keep_columns(self, columns):
        for col in self.df.columns:
            if col not in columns:
                self.drop_column(col)

    def drop_column(self, name):
        if name in self.df.columns:
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


    def save_columns(self):
        for name in self.df.columns:
            index = self.df.columns.get_loc(name)
            if not name in self.column_dict:
                self.column_dict[name] = index
            else:
                if index != self.column_dict[name]:
                    print("INVALID COLUMN INDEX")