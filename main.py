import pandas as pd

from stock import *
from aggregator import *
from sequencer import *
from scaler import *
from target import *
from features import *
from predictor import *
from samplegenerator import *
from feargreed import *
from plotter import *

pd.options.display.float_format = '{:.10}'.format
np.set_printoptions(suppress=True, precision=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#set_seed(112)


epochs = 200
test_size = 0.25
sequence_length = 15
reload = False
train = False
plot = Plotter()
sequencer = Sequencer(sequence_length)


# stock_names = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'TSLA', 'WAC.DE', 'VZ']
stock_name = '^NDX'
# start_date = '2010-01-01'
# start_date = '2017-01-01'
start_date = '2020-01-01'
end_date = None  # '2023-06-01'
reload = True
train = True

fg = FearAndGreed()
fg.load('data/feargreed_simple.json')
fg.map_ratings()
#fg.show_min_max()

scaler = Scaler()
scaler.scale_same(['Open', 'Close', 'High', 'Low', 'BM', 'BU',
                  'BL', 'SMA20', 'SMA50', 'Adj Close'])
scaler.scale_same(['A', 'B'])
scaler.scale_same(['MACD', 'MACD_Signal', 'MACD_Diff'])
scaler.scale_once_factor('RSI', 0.01)
scaler.scale_once_factor('fear_and_greed_historical_val', 0.01)
scaler.scale_once_minmax('Volume')
scaler.scale_once_minmax('stock_price_strength_val')
scaler.scale_once_minmax('stock_price_breadth_val')
scaler.scale_once_minmax('put_call_options_val')
scaler.scale_once_minmax('market_volatility_vix_val')
scaler.scale_once_minmax('market_volatility_vix_50_val')
scaler.scale_once_minmax('junk_bond_demand_val')
scaler.scale_once_minmax('safe_haven_demand_val')

s = Stock()
s.load(stock_name, start_date=start_date, end_date=end_date, reload=reload)
a = Aggregator(s)
fg.add(a)
features = Features(a)
features.add_bollinger()
features.add_rsi()
features.add_macd()
features.add_sma()
# features.add_ema()
# features.add_atr()
features.add_dummy()
Target(a).add_target2()
keep = [
    #'Open', 'High', 'Low',
    'Close',
     #'A', #'B',
    # 'Adj Close', 'Volume',
     'fear_and_greed_historical_val',
    # 'fear_and_greed_historical_rating',
    # 'market_momentum_sp500_val', 'market_momentum_sp500_rating',
    # 'market_momentum_sp125_val', 'market_momentum_sp125_rating',
    # 'stock_price_strength_val', 'stock_price_strength_rating',
    # 'stock_price_breadth_val', 'stock_price_breadth_rating',
     'put_call_options_val',
    # 'put_call_options_rating',
    # 'market_volatility_vix_val', 'market_volatility_vix_rating',
    # 'market_volatility_vix_50_val', 'market_volatility_vix_50_rating',
    # 'junk_bond_demand_val', 'junk_bond_demand_rating',
    # 'safe_haven_demand_val', 'safe_haven_demand_rating', 'BM', 'BU', 'BL',
     'RSI',
    # 'MACD',
     'MACD_Signal',
    # 'MACD_Diff', 'SMA20', 'SMA50',
    'Target'
]
a.keep_columns(keep)

#a.drop_not_needed_stock_columns()
#a.drop_not_needed_fear_and_greed_columns()
a.save_columns()
a.drop_nan()
scaler.scale_df_once(a)
scaler.scale_df(a)

plot.show(a.df, features=None, marker='Target', block=True)

#plot.show(a.df, features=['Close', 'A', 'B'], marker='Target', block=False)


scaler.resolve_names(a.column_dict)
sequencer.add(a.df)
scaler.process_sequences(sequencer.X)
sequencer.split_train_test(test_size)


predictor = Predictor(sequence_length=sequence_length, sequencer=sequencer)
predictor.model7()

if train:
    predictor.train(epochs=epochs, save=True)
    #predictor.optimize(model=predictor.model7, save=True)
else:
    predictor.load()

# predictor.confusion(sequencer.get_data_train())
# predictor.predict(s.df, sequencer.get_data_train())
predictor.confusion(sequencer.get_data_test(), show=True, block=False)
predictor.predict(a.df, sequencer.get_data_test())


# plot.show(s.df, features=['Close', 'Low', 'High', 'Target', 'y'], markers=True)
plot.show(a.df, features=['Close'], marker='Target', marker2='y')


pass
