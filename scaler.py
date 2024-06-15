from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np


class Scaler:
    def __init__(self):
        self.scale_factor_dict = {}
        self.scale_factor_index_dict = {}
        self.scale_same_name_list = []
        self.scale_same_index_list = []

        pass

    def scale_same(self, names):
        self.scale_same_name_list.append(names)
        pass

    def scale_factor(self, name, factor):
        self.scale_factor_dict[name] = factor
        pass

    def scale_all(self):
        self.scale_same(["Open", "Close", "High", "Low", "SMA20", "SMA50"])#, "SMA100", "SMA200"])
        self.scale_same(["Volume"])
        self.scale_same(["EMA"])
        self.scale_same(["ATR"])
        self.scale_same(["MACD", "MACD_Signal"])
        self.scale_same(["MACD_Diff"])
        self.scale_factor("RSI", 0.01)        

    def resolve_names(self, column_dict):
        for name, factor in self.scale_factor_dict.items():
            col_index = column_dict.get(name, None)
            if col_index is not None:
                self.scale_factor_index_dict[col_index] = factor

        for scale_names in self.scale_same_name_list:
            columns = []
            for name in scale_names:
                col_index = column_dict.get(name, None)
                if col_index is not None:
                    columns.append(col_index)
            self.scale_same_index_list.append(columns)


    def process_sequences(self, sequences):
        for sequence in sequences:
            self.process_sequence(sequence)


    def process_sequence(self, sequence):
        for col, factor in self.scale_factor_index_dict.items():
            sequence[:, col] *= factor

        for columns in self.scale_same_index_list:
            scaler = MinMaxScaler(feature_range=(0, 1))
            minmax_values = []
            for col in columns:
                column_data = sequence[:, col]
                minmax_values.append(column_data.min())
                minmax_values.append(column_data.max())
                #filtered_column_data = column_data[~np.isnan(column_data)]
                #if len(filtered_column_data) > 0:
                #    minmax_values.append(np.min(filtered_column_data))
                #    minmax_values.append(np.max(filtered_column_data))
                #minmax_values.append(np.nanmin(sequence[:, col]))
                #minmax_values.append(np.nanmax(sequence[:, col]))
            if len(minmax_values) > 0:
                minmax = pd.DataFrame(minmax_values)
                # print(minmax)
                scaler.fit(minmax)

                for col in columns:
                    sequence[:, [col]] = scaler.transform(sequence[:, [col]])
        pass
