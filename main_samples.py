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


epochs = 50
test_size=0.2
sequence_length = 20

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(112)

target = Target()
features = Features()
sequencer = Sequencer(sequence_length)

plot = Plotter()
s = SampleGenerator()
s.generate(start='2015-01-01')
target.add_target(s.df, distance=1, target_percent=0.1)
sequencer.add(s.df)
features.save_columns(s.df)

#print(sg.df.head())
#plot.show(sg.df, features=['Close', 'Low', 'High', 'Target'])
#exit(0)


scaler = Scaler()
scaler.scale_same(["Close", "High", "Low"])

scaler.resolve_names(features.column_dict)
scaler.process_sequences(sequencer.X)

sequencer.split_train_test(test_size)


predictor = Predictor(sequence_length=sequence_length, sequencer=sequencer)

#predictor.optimize(save=True)

predictor.model7()
#predictor.train(epochs=epochs, save=True)

predictor.load()
predictor.confusion(sequencer.get_data_all())
predictor.predict(s.df, sequencer.get_data_all())
plot.show(s.df, features=['Close', 'Low', 'High', 'Target', 'y'])


  

pass
