import pandas as pd

from stock import *
from sequencer import *
from scaler import *
from target import *
from features import *
from predictor import *
from samplegenerator import *
from plotter import *

pd.options.display.float_format = '{:.10}'.format
np.set_printoptions(suppress=True, precision=2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(112)


epochs = 300
test_size = 0.05
sequence_length = 10
reload = False
train = False
plot = Plotter()
target = Target()
sequencer = Sequencer(sequence_length)


#    stock_names = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'TSLA']
#stock_names = ['MSFT']
stock_name = '^NDX'
#stock_name = 'WAC.DE'
#stock_name = 'VZ'
start_date = '2010-01-01'
end_date = None # '2023-06-01'
#reload = True
train = True

s = Stock()
s.load(stock_name, start_date=start_date, end_date=end_date, reload=reload)
s.preprocess()
features = Features(s.df)
features.add_bollinger()
features.add_rsi()
features.save_columns()
s.add_target()
s.drop_nan()
#s.apply_pivot(dist=5)
#s.apply_pointpos()
#s.head()
#print(s.df[100:150])
#plot.show_support(s.df, bin_width=10)

plot.show(s.df, features=['Close', 'RSI', 'BM', 'BU', 'BL'], marker='Target', block=True)


scaler = Scaler()
scaler.scale_same(["Open", "Close", "High", "Low", 'BM', 'BU', 'BL'])
scaler.scale_factor('RSI', 0.01)

scaler.resolve_names(features.column_dict)
sequencer.add(s.df)
scaler.process_sequences(sequencer.X)
sequencer.split_train_test(test_size)



#s = SampleGenerator()
#s.generate(start='2015-01-01')
#target.add_target(s.df, distance=1, target_percent=0.1)
#sequencer.add(s.df)
#features.save_columns(s.df)

#print(sg.df.head())
#plot.show(sg.df, features=['Close', 'Low', 'High', 'Target'])
#exit(0)


#scaler = Scaler()
#scaler.scale_same(["Close", "High", "Low"])

#scaler.resolve_names(features.column_dict)
#scaler.process_sequences(sequencer.X)

#sequencer.split_train_test(test_size)


predictor = Predictor(sequence_length=sequence_length, sequencer=sequencer)

#predictor.optimize(model=predictor.model7_opt, save=True)

predictor.model7()
if train:
    predictor.train(epochs=epochs, save=True)
else:
    predictor.load()

#predictor.confusion(sequencer.get_data_train())
#predictor.predict(s.df, sequencer.get_data_train())
predictor.confusion(sequencer.get_data_test(), show=True, block=False)
predictor.predict(s.df, sequencer.get_data_test())


#plot.show(s.df, features=['Close', 'Low', 'High', 'Target', 'y'], markers=True)
plot.show(s.df, features=['Close'], marker='y')


  

pass
