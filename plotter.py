import matplotlib.pyplot as plt
import pandas as pd
import mplcursors


class Plotter:
    def __init__(self):
        pass

    def show(self, df: pd.DataFrame, features=None, marker=None, marker2=None, block=True):
        plt.figure(figsize=(20, 6))

        if features is None:
            features = df.columns

        for f in features:
            if f in df.columns and f != 'Target':
                plt.plot(df[f], label=f)


        offset = 0.000
        if marker2 is not None:
            pivot1 = df.index[df[marker2] == 1]
            pivot2 = df.index[df[marker2] == 2]
            plt.plot(pivot1, df['Close'][pivot1]+offset, '^', markerfacecolor='none', markeredgecolor='green', markeredgewidth=1, markersize=8, label=f'{marker2}_buy')
            plt.plot(pivot2, df['Close'][pivot2]+offset, 'v', markerfacecolor='none', markeredgecolor='red', markeredgewidth=1, markersize=8, label=f'{marker2}_sell')

        if marker is not None:
            pivot1 = df.index[df[marker] == 1]
            pivot2 = df.index[df[marker] == 2]
            plt.plot(pivot1, df['Close'][pivot1], 'g^', markersize=5, label=f'{marker}_buy')
            plt.plot(pivot2, df['Close'][pivot2], 'rv', markersize=5, label=f'{marker}_sell')

        plt.legend()
        plt.show(block=block)


    def show2(self, df: pd.DataFrame, features=None, marker=None, block=True):
        fig, ax = plt.subplots()
        lines = []
        if features is None:
            features = df.columns

        for f in features:
            if f in df.columns and f != 'Target':
                line, = ax.plot(df[f], label=f)
                lines.append(line)

        leg = ax.legend(loc='upper right')
        cursor = mplcursors.cursor(leg, hover=True)

        # Make the legend interactive
        #lined = {}  # Will map legend lines to original lines
        #for legline, origline in zip(leg.get_lines(), lines):
        #    legline.set_picker(True)  # Enable picking on the legend line
        #    lined[legline] = origline

        # Define the function to toggle visibility on click
        def on_click(legend_entry):
            origline = legend_entry.artist
            vis = not origline.get_visible()
            origline.set_visible(vis)
            alpha = 1.0 if vis else 0.2
            legend_entry.annotation.set_alpha(alpha)
            fig.canvas.draw()

        # Connect the cursor object to the click event
        cursor.connect("add", on_click)

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
