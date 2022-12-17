import numpy as np

class dataset:
    def __init__(self, batch_size=64, X=None, Y=None):
        self.max_num = len(X)//batch_size
        self.index = 0
        self.X = X
        self.Y = Y
        self.item = zip(X, Y)
        self.batch_size = batch_size

    def __iter__(self):
        return self
        
    def __next__(self):
        self.index += 1
        if self.index < self.max_num:
            index = np.random.choice(len(self.X), size=self.batch_size, replace=False)
            return self.X[index], self.Y[index]
        else:
            raise StopIteration