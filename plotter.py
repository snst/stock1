import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(self):
        pass

    def show(self, df: pd.DataFrame, features=["Close"], marker=None, block=True):
        plt.figure(figsize=(20, 6))

        for f in features:
            if f in df.columns:
                plt.plot(df[f], label=f)

        if marker is not None:
            pivot1 = df.index[df[marker] == 1]
            pivot2 = df.index[df[marker] == 2]
            plt.plot(pivot1, df['Close'][pivot1], 'gx', label='buy')
            plt.plot(pivot2, df['Close'][pivot2], 'rx', label='sell')

        plt.legend()
        plt.show(block=block)

    def show_support(self, df, bin_width=1):
        dfkeys = df[:]

        # Filter the dataframe based on the pivot column
        high_values = dfkeys[dfkeys['Pivot'] == 2]['High']
        low_values = dfkeys[dfkeys['Pivot'] == 1]['Low']

        # Calculate the number of bins
        bins = int((high_values.max() - low_values.min()) / bin_width)

        # Create the histograms
        plt.figure(figsize=(10, 5))
        plt.hist(high_values, bins=bins, alpha=0.5,
                 label='High Values', color='red')
        plt.hist(low_values, bins=bins, alpha=0.5,
                 label='Low Values', color='blue')

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of High and Low Values')
        plt.legend()
        plt.show(block=True)
