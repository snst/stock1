import numpy as np


class Target:
    def __init__(self):
        pass

    def add_target(self, df, source='Close', distance=5, target_percent=2):
        values = df[source].values
        target = np.zeros(len(values))
        n_buy_signal = 0
        n_nobuy_signal = 0

        for i in range(len(values)-distance-1):
            percent = (100.0 / values[i]
                       * values[i+distance]) - 100.0
            buy = percent >= target_percent
            # buy = values[i] < values[i+distance]
            target[i] = 1 if buy else 0
            if buy:
                n_buy_signal += 1 
            else:
                n_nobuy_signal += 1
        
        print(f'n_buy_signal: {n_buy_signal}, n_nobuy_signal: {n_nobuy_signal}')
            
        df['Target'] = target
#        df = df.head(len(df) - distance)
        df.drop(df.index[-distance:], inplace=True)
        pass



"""
    def add_target_avg(self, source='Close'):
        F = 1
        P = 3
        source_feature = self.data[source].values
        target = np.zeros(len(source_feature))

        for i in range(len(source_feature)-F-P-1):
            sum = 0
            for k in range(i+F, i+F+P):
                sum += source_feature[k]
            avg = sum/P
            target[i] = 1 if avg > source_feature[i] else 0

        self.data['Target'] = target
        # df = df_target.iloc[:-(F+P)]
"""