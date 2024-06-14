import random
import pandas as pd
import numpy as np


class SampleGenerator:
    def __init__(self):
        pass

    def generate(self, start='2018-01-01', end='2020-12-31'):
        date_range = pd.date_range(
            start, end=end, freq='D')

        # Create a DataFrame with the date range as the index
        df = pd.DataFrame(index=date_range)

        # Generate random values for column 'a'
        #np.random.seed(42)  # For reproducibility
        df['Close'] = 0
        df['Low'] = 0
        df['High'] = 0
        #df['Date'] = df.index
        df['Target'] = 0

        val = 5.1
        n = len(df)
        next_low = 10
        next_high = 5
        last_low = None
        last_high = None

        for i in range(n):
            #val += + random.uniform(-0.2, 0.2 )

            if i-1 == last_high and i-2 == last_low:
                val += 1

            df.iloc[i, df.columns.get_loc('Close')] = val
            high = val + 1
            low = val - 1
            if i == next_low:
                last_low = i
                low = val - 2
                next_low += int(random.uniform(5, 12))
                if random.uniform(0, 1) > 0.4:
                    next_high = i + 1
            if i == next_high:
                last_high = i
                high = val + 2
                next_high += int(random.uniform(3, 12))

    
            df.iloc[i, df.columns.get_loc('Low')] = low
            df.iloc[i, df.columns.get_loc('High')] = high

            #df.iloc[i, df.columns.get_loc('Low')] = val + random()

        self.df = df
        pass

        # Set a constant value of 4 for column 'b'
#        self.df['Low'] = np.random.rand(len(date_range))
