import numpy as np

def Sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-7))

def Softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def Relu(z):
    return np.maximum(z, 0)

def cross_entropy(Y, Y_prediction):
    return -np.mean(np.multiply(Y, np.log(Y_prediction + 1e-07)))