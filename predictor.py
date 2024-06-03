import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

class Predictor:
    def __init__(self):
        self.model = None
        self.history = None
        pass

    def model1(self, x_train, sequence_length):
        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=False, input_shape=(sequence_length, x_train.shape[-1])))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model2(self, x_train, sequence_length):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(sequence_length, x_train.shape[-1]), return_sequences=False))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def model3(self, x_train, sequence_length):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(sequence_length, x_train.shape[-1]), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        self.model.summary()

    def train(self, x_train, y_train, epochs=300, batch_size=32, validation_split=0.2):
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # Train the model
        #history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.show()


    def confusion(self, x_input, y_input):
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(x_input, y_input)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')

        val = 0.5
        while val < 1:
            print(f"val:{val}")
            # Make predictions
            y_pred_prob = self.model.predict(x_input)
            y_pred = (y_pred_prob > val).astype(int)

            # Evaluate predictions
            print(classification_report(y_input, y_pred))
            conf_matrix = confusion_matrix(y_input, y_pred)

            # Plot confusion matrix
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            val += 0.1
            break

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)