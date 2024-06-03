import yfinance as yf
import pandas as pd
import os

from stock import *
from sequencer import *
from scaler import *
from target import *
from features import *
from predictor import *

pd.options.display.float_format = '{:.10}'.format
np.set_printoptions(suppress=True, precision=2)

sequence_length = 15


predict = False
#predict = True
model_name = 'model3.model'
epochs = 150


if not predict:
    stock_names = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'TSLA']
    start_date = '2019-01-01'
    test_size=0.2

else:
    stock_names = ['MSFT']
    stock_names = ['META']
    start_date = '2023-01-01'
    test_size=0.9


#stock_names = ['MSFT']#, 'AAPL', 'AMZN', 'GOOGL', 'TSLA']

end_date = None # '2023-06-01'
reload = False
#reload = True

features = Features()
target = Target()
sequencer = Sequencer(sequence_length)


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


scaler = Scaler()
scaler.scale_same(["Open", "Close", "High", "Low", "SMA20", "SMA50"])#, "SMA100", "SMA200"])
scaler.scale_same(["Volume"])
scaler.scale_same(["MACD", "MACD_Signal"])
scaler.scale_same(["MACD_Diff"])
scaler.scale_factor("RSI", 0.01)

scaler.resolve_names(features.column_dict)

scaler.process_sequences(sequencer.x)
sequencer.split_train_test(test_size)

predictor = Predictor()
if not predict:
    predictor.model3(sequencer.x_train, sequence_length)
    predictor.train(sequencer.x_train, sequencer.y_train, epochs=epochs)
    predictor.save(model_name)
else:
    predictor.load(model_name)

predictor.confusion(sequencer.x_test, sequencer.y_test)

pass
