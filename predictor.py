import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras_tuner as kt # pip install keras-tuner
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, ConvLSTM1D
from tensorflow.keras.layers import concatenate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

class Predictor:
    def __init__(self, sequence_length, sequencer, name="model"):
        self.name = name
        self.model = None
        self.history = None
        self.sequence_length = sequence_length
        self.sequencer = sequencer
        self.feature_length = self.sequencer.X.shape[-1]
        self.input_shape = (self.sequence_length, self.feature_length)
        pass

    def model1(self):
        self.name = 'model1'
        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=False, input_shape=(self.sequence_length, self.feature_length)))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model2(self):
        self.name = 'model2'
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.sequence_length, self.feature_length), return_sequences=False))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model3(self):
        self.name = 'model3'
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.sequence_length, self.feature_length), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model4(self):
        self.name = 'model4'
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, self.feature_length)))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model7(self):
        self.name = 'model7'
        self.model = Sequential()
        self.model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        self.model.add(Dropout(0.2))
        #self.model.add(keras.layers.MaxPooling1D(pool_size=3))
        self.model.add(keras.layers.LSTM(50, return_sequences=True))
        self.model.add(keras.layers.LSTM(50))
        #self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.add(Dense(50, activation='relu'))  
        self.model.add(keras.layers.Dense(3, activation='softmax'))
#        self.model.add(keras.layers.Dense(3, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model7_opt(self, hp):
        self.name = 'model7'
        hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
        hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 11])
        hp_pool_size = hp.Choice('pool_size', values=[2, 3, 5])
        hp_lstm1_units = hp.Int('lstm1_units', min_value=10, max_value=100, step=10)        
        hp_lstm2_units = hp.Int('lstm2_units', min_value=10, max_value=100, step=10)        
        self.model = Sequential()
        self.model.add(keras.layers.Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', input_shape=self.input_shape))
        self.model.add(keras.layers.MaxPooling1D(pool_size=hp_pool_size))
        self.model.add(keras.layers.LSTM(hp_lstm1_units, return_sequences=True))
        self.model.add(keras.layers.LSTM(hp_lstm2_units))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        #self.model.summary()
        return self.model

    def model4_opt(self, hp):
        self.name = 'model4_opt'
        # Hyperparameters to tune: number of filters, kernel size, and number of Conv1D layers
        hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
        hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 11])
        hp_pool_size = hp.Choice('pool_size', values=[2, 3])
        hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)        
        self.model = Sequential()
        self.model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', input_shape=(self.sequence_length, self.feature_length)))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(pool_size=hp_pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(hp_dense_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                            loss='binary_crossentropy', metrics=['accuracy'])        
#        self.model.summary()
        return self.model


    def model6(self):
        self.name = 'model6'
        self.model = Sequential()
        #self.model.add(LSTM(50, input_shape=(self.sequence_length, self.feature_length), return_sequences=False))
        self.model.add(ConvLSTM1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.sequence_length, 1, self.feature_length)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model5(self):
        self.name = 'model5'
        inputs1 = Input(shape=(self.sequence_length, self.feature_length))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # head 2
        inputs2 = Input(shape=(self.sequence_length, self.feature_length))
        conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # head 3
        inputs3 = Input(shape=(self.sequence_length, self.feature_length))
        conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(100, activation='relu')(merged)
        outputs = Dense(1, activation='sigmoid')(dense1)
        self.model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()


    def optimize(self, model, save=False):

        # Instantiate the tuner
        tuner = kt.Hyperband(model,
                            objective='val_accuracy',
                            max_epochs=50,
                            factor=3,
                            directory='my_dir',
                            project_name='cnn_tuning')

        # Define early stopping
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Run the hyperparameter search
        tuner.search(self.sequencer.X, self.sequencer.y, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print("The hyperparameter search is complete:")
        print(best_hps)
        #print(f"""
        #The hyperparameter search is complete. The optimal number of filters is {best_hps.get('filters')},
        #the optimal kernel size is {best_hps.get('kernel_size')}, the optimal pool size is {best_hps.get('pool_size')},
        #the optimal number of dense units is {best_hps.get('dense_units')}, and the optimal learning rate is {best_hps.get('learning_rate')}.
        #""")

        # Build the model with the optimal hyperparameters and train it
        self.model = tuner.hypermodel.build(best_hps)
        history = self.model.fit(self.sequencer.X, self.sequencer.y, epochs=50, validation_split=0.2)
        if save:
            self.save()



    def train(self, epochs=300, batch_size=32, validation_split=0.2, save=False):
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # Train the model
        #history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        self.history = self.model.fit(self.sequencer.X_train, self.sequencer.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        #self.history = self.model.fit([X_train,X_train,X_train], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.show(block=True)
        if save:
            self.save()


    def confusion(self, input, show=True, block=False):
        X, y_true_enc, d = input
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(X, y_true_enc)
        #test_loss, test_accuracy = self.model.evaluate([x_input,x_input,x_input], y_input)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')

        # Make predictions
        y_pred_enc = self.model.predict(X)
        #y_pred_prob = self.model.predict([x_input, x_input, x_input])
        #y_pred = (y_pred_prob > 0.5).astype(int)
        y_pred = np.argmax(y_pred_enc, axis=1)
        y_true = np.argmax(y_true_enc, axis=1)

        report = classification_report(y_true, y_pred, target_names=['wait', 'buy', 'sell'])
        print(report)

        if show:
            conf_matrix = confusion_matrix(y_true, y_pred)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show(block=block)

    def save(self):
        self.model.save(self.name)

    def load(self, name=None):
        if name:
            self.name = name
        self.model = tf.keras.models.load_model(self.name)


    def predict(self, df, input):
        X, y, d = input
        len_df = len(df)
        len_X = len(X)
        df['y'] = 0

        y_pred_enc = self.model.predict(X)
        y_pred = np.argmax(y_pred_enc, axis=1)
        i = 0
        for dindex in d:
            df.loc[dindex, 'y'] = y_pred[i]
            i += 1
            pass
        pass    