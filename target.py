import numpy as np
import myutils


class Target:
    def __init__(self, data):
        self.data = data
        pass

    def add_target(self, distance=5, percent=1):
        df = self.data.df
        BUY = 1
        SELL = 2
        l = len(df)
        target = np.zeros(l)
        for i in range(0, l-distance):
            I = df.Close.iloc[i]
            for k in range(0, distance):
                K = df.Close.iloc[i+k]
                if myutils.percent(I, K) < -0.1:
                    break
                if myutils.percent(I, K) > percent:
                    target[i] = BUY
                    break
            for k in range(0, distance):
                K = df.Close.iloc[i+k]
                if myutils.percent(I, K) < -percent:
                    target[i] = SELL
                    break
        df['Target'] = target
        self.data.df = df
        pass

    def add_target2(self, distance=1, percent=1):
        df = self.data.df
        BUY = 1
        SELL = 2
        l = len(df)
        target = np.zeros(l)
        for i in range(0, l-distance):
            I = df.Close.iloc[i]
            K = df.Close.iloc[i+1]
            if myutils.percent(I, K) > percent:
                target[i] = BUY
            elif myutils.percent(I, K) < -percent:
                target[i] = SELL
        df['Target'] = target
        self.data.df = df
        pass

    def pivotid(self, df1, time_index, n1, n2): #n1 n2 before and after candle l
        #https://colab.research.google.com/drive/1ATNIwG-gYUHs3BfHrsyfeS7ctTxBGW1t#scrollTo=964219a4
        #https://www.youtube.com/watch?v=MkecdbFPmFY
        l = df1.index.get_loc(time_index)
        if l-n1 < 0 or l+n2 >= len(df1):
            return 0
        
        pividlow=1
        pividhigh=1
        first_i = l-n1
        last_i = l+n2
        for i in range(l-n1, l+n2+1):
#            if(df1.Low.iloc[l]>df1.Low.iloc[i]):
            if(df1.Close.iloc[l]>df1.Close.iloc[i]):
                pividlow=0
#            if(df1.High.iloc[l]<df1.High.iloc[i]):
            if(df1.Close.iloc[l]<df1.Close.iloc[i]):
                pividhigh=0
            """
            if i == first_i and pividlow:
                percent = (100 / df1.Close.iloc[l] * df1.Close.iloc[i]) - 100
                if percent < 1:
                    pividlow=0
                pass
            if i == last_i and pividhigh:
                percent = (100 / df1.Close.iloc[l] * df1.Close.iloc[i]) - 100
                if percent < -1:
                    pividhigh=0
                pass
            """
        if pividlow and pividhigh:
            return 3
        elif pividlow:
            return 1
        elif pividhigh:
            return 2
        else:
            return 0
        
    def apply_pivot(self, dist=10):
        df = self.data.df
        df['Pivot'] = df.apply(lambda x: self.pivotid(df, x.name, dist, dist), axis=1)
        self.data.df = df
