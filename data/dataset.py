import numpy as np
from scipy.io import loadmat
from util.util import to_categorical
from sklearn.model_selection import train_test_split

class dataset:
    def __init__(self, X=None, Y=None, batch_size=256):
        self.max_num = len(X)//batch_size
        self.index = -1
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.X)
        
    def __next__(self):
        if self.index < self.max_num:
            if self.batch_size == 1:
                index = self.index
            else:
                index = np.random.choice(len(self.X), size=self.batch_size, replace=False)
            self.index += 1         
            return np.squeeze(self.X[index]), self.Y[index]
        else:
            self.index = 0
            raise StopIteration
        
def get_mnist():
    mnist = loadmat('./data/mnist-original.mat')
    x = mnist['data'].T
    x = (x/255.).astype('float32')
    y = mnist['label'].T.flatten()
    y = to_categorical(y)
    return train_test_split(x, y, test_size=0.15, random_state=42)