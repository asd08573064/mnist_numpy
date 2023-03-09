import numpy as np

def Sigmoid(z, derivative=False):
    if derivative:
        return np.exp(-z)/(1+np.exp(-z))**2
    else:
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-7))
    
def Relu(z, derivative=False):
    if derivative:
        z = np.where(z < 0, 0, z)
        return np.where(z >= 0, 1, z)
    else:
        return np.maximum(z, 0)

def Softmax(x, derivative=False):
    if derivative:
        return x*(1.-x+1e-07)
    else:
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

def Cross_Entropy(Y, Y_prediction, derivative=False):
    # print(Y.shape, Y_prediction.shape)
    if derivative:
        return (Y-Y_prediction)/(Y*(1.-Y) + 1e-07)
    else:  
        return -np.mean(np.multiply(Y, np.log(Y_prediction + 1e-07)))