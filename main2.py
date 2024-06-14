import yfinance as yf
import pandas as pd
import os

from stock import *
from sequencer import *
from scaler import *
from target import *
from features import *
#from predictor import *
from samplegenerator import *
from plotter import *

pd.options.display.float_format = '{:.10}'.format
np.set_printoptions(suppress=True, precision=2)

sequence_length = 30


predict = False
predict = True
model_name = 'model4.model'
epochs = 300
test_size=0.2

target = Target()
features = Features()
sequencer = Sequencer(sequence_length)

plot = Plotter()
s = SampleGenerator()
s.generate()
target.add_target(s.df, distance=1, target_percent=0.1)
sequencer.add(s.df)
features.save_columns(s.df)

#print(sg.df.head())
#plot.show(sg.df, features=['Close', 'Low', 'High', 'Target'])
#exit(0)

#if not predict:
#    stock_names = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'TSLA']
#    #stock_names = ['MSFT']
#    start_date = '2019-01-01'
#else:
#    stock_names = ['ORCL']
#    #stock_names = ['META']
#    start_date = '2023-01-01'
#    test_size=0.9


#stock_names = ['MSFT']#, 'AAPL', 'AMZN', 'GOOGL', 'TSLA']

end_date = None # '2023-06-01'
reload = False
#reload = True




#load_stocks(stock_names, start_date, end_date, reload)

scaler = Scaler()
#scaler.scale_all()
scaler.scale_same(["Open", "Close", "High", "Low", "SMA20", "SMA50"])#, "SMA100", "SMA200"])

scaler.resolve_names(features.column_dict)
scaler.process_sequences(sequencer.X)

sequencer.split_train_test(test_size)

predictor = Predictor()
if not predict:
    predictor.model5(sequencer.X_train, sequence_length)
    predictor.train(sequencer.X_train, sequencer.y_train, epochs=epochs)
    predictor.save(model_name)
else:
    predictor.load(model_name)

predictor.confusion(sequencer.X_test, sequencer.y_test)

pass
