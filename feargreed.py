import pandas as pd
import json


class FearAndGreed:
    FAG_MAPPING = {
        'extreme fear': 0.0,
        'fear': 0.25,
        'neutral': 0.5,
        'greed': 0.75,
        'extreme greed': 1.0,
    }

    FAG_RATINGS = ['fear_and_greed_historical_rating', 'market_momentum_sp500_rating', 'market_momentum_sp125_rating', 'stock_price_strength_rating',
                   'stock_price_breadth_rating', 'put_call_options_rating', 'market_volatility_vix_rating', 'market_volatility_vix_50_rating', 'junk_bond_demand_rating', 'safe_haven_demand_rating']

    def __init__(self):
        self.df = None
        pass

    def load(self, path):
        with open(path) as file:
            data = json.load(file)

        self.df = pd.DataFrame.from_dict(data, orient='index')
        self.df.index = pd.to_datetime(self.df.index)
        pass

    def map_ratings(self):
        for col in FearAndGreed.FAG_RATINGS:
            self.df[col] = self.df[col].map(FearAndGreed.FAG_MAPPING)
        pass

    def add(self, data):
        for col in self.df.columns:
            data.df[col] = self.df[col]
            pass
        pass

    def show_min_max(self):
        for col in self.df.columns:
            print(f'{col}\t{self.df[col].min()}\t{self.df[col].max()}')
