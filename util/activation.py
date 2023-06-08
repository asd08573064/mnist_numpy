import numpy as np
import pdb

def Sigmoid(z, derivative=False):
    if derivative:
        return np.exp(-z)/(1+np.exp(-z))**2
    else:
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-7))
    
def relu(z, derivative=False):
    if derivative:
        z = np.where(z < 0, 0, z)
        z = np.where(z >= 0, 1, z)
        return z
    else:
        return np.maximum(0, z)
    
def leakyrelu(z, derivative=False):
    if derivative:
        z = np.where(z < 0, 0.2, z)
        z = np.where(z >= 0, 1, z)
        return z
    else:
        return np.maximum(0.2*z, z)

def Softmax(x, derivative=False):
    if derivative:
        return x*(1.-x+1e-07)
    else:
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
def Tanh(x, derivative=True):
    exps = np.clip(np.exp(x), 1e-8, 1 - (1e-7))
    n_exps = np.clip(np.exp(-x), 1e-8, 1 - (1e-7))
    a  = (exps - n_exps) / (exps + n_exps)
    if derivative:
        return 1-a**2
    else:
        return a
        
def BinaryCrossEntropy(Y, y_pred, derivative=False):
    if derivative:
        return (Y-y_pred)/(Y*(1.-Y) + 1e-07)
    else:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1-Y) * np.log(1-y_pred + 1e-7)
        term_1 = Y * np.log(y_pred + 1e-7)
        return -np.mean(term_0+term_1)

def Cross_Entropy(Y, y_pred, derivative=False):
    if derivative:
        return (Y-y_pred)/(Y*(1.-Y) + 1e-07)
    else:  
        return -np.mean(np.multiply(Y, np.log(y_pred + 1e-07)))