import numpy as np
from scipy.io import loadmat
from util.util import to_categorical
from sklearn.model_selection import train_test_split

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
            if self.batch_size == 1:
                index = self.index
            else:
                index = np.random.choice(len(self.X), size=self.batch_size, replace=False)
            return self.X[index], self.Y[index]
        else:
            raise StopIteration
        
def get_mnist():
    mnist = loadmat('./data/mnist-original.mat')
    x = mnist['data'].T
    x = (x/255.).astype('float32')
    y = mnist['label'].T.flatten()
    y = to_categorical(y)
    return train_test_split(x, y, test_size=0.15, random_state=42)