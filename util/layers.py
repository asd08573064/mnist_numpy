import numpy as np
import util.activation as A

class Dropout():
    def __init__():
        pass
    def forward(self, x):
        pass
    def backward(self, x):
        pass
    
class ReLU():
    def __init__(self):
        self.output = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        self.output = A.Relu(x)
        return self.output
    
    def backward(self, x, grad=None):
        if grad is not None:
            return grad * A.Relu(x, derivative=True)
        return A.Relu(x, derivative=True)
    
class Softmax():
    def __init__(self):
        self.output = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        self.output = A.Softmax(self.input)
        return self.output
    
    def backward(self, grad=None):
        if grad is not None:
            return grad * A.Softmax(self.output, derivative=True)
        return A.Softmax(self.output, derivative=True)
    
class Sigmoid():
    def __init__(self):
        self.output = None
        self.input = None
    def forward(self, x):
        self.input = x
        self.output = A.Sigmoid(x)
        return self.output
    
    def backward(self, grad=None):
        if grad is not None:
            return grad * A.Sigmoid(self.input, derivative=True)
        return A.Sigmoid(self.input, derivative=True)
    
class Cross_Entropy():
    def __init__(self):
        self.output = None
        self.input = None
    def forward(self, Y, Y_prediction):
        self.output = A.Cross_Entropy(Y, Y_prediction)
        self.input = Y
        return self.output
    
    def backward(self, Y_prediction, grad=None):
        if grad is not None:
            return grad * A.Cross_Entropy(self.input, Y_prediction, derivative=True)
        return A.Cross_Entropy(self.input, Y_prediction, derivative=True)

class Batch_Norm():
    def __init__(self):
        self.gamma = 0
        self.beta = 0
        self.cache = {
            'd_gamma_cache' : 1,
            'd_beta_cache' : 0
        }
        self.running_mu = .0
        self.running_var = 1.
        self.input = None
        
    def forward(self, x, is_training=True, Beta=0.1):
        eps = 1e-9
        self.input = x
        if is_training:
            self.mu = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.x_hat = (x - self.mu) / np.sqrt(self.var + eps)
            self.y = self.gamma * self.x_hat + self.beta
            
            self.running_mu = Beta * self.running_mu + (1. - Beta) * self.mu
            self.running_var = Beta * self.running_var + (1. - Beta) * self.var
            
        else:
            self.x_hat = (x - self.running_mu) / np.sqrt(self.running_var + eps)
            self.y = self.gamma * self.x_hat + self.beta
        
        return self.y
    
    def backward(self, da):
        eps = 1e-9
        batch_size = da.shape[0]
        self.cache['d_beta'] = np.sum(da, axis=0)
        self.cache['d_gamma'] = np.sum(da * self.x_hat, axis=0)
        d_x_hat = self.gamma * da
        self.cache['d_x'] = (1./batch_size) * (batch_size * d_x_hat - np.sum(d_x_hat, axis=0) - self.x_hat * np.sum(d_x_hat * self.x_hat, axis=0))/(batch_size * np.sqrt(self.var + eps))
        return self.cache['d_x']
    
    def update(self, learning_rate, Beta=0.95):
        self.beta = self.beta - learning_rate * (Beta*self.cache['d_beta_cache'] + (1-Beta) * self.cache['d_beta'])
        self.gamma = self.gamma - learning_rate * (Beta*self.cache['d_gamma_cache'] + (1-Beta) * self.cache['d_gamma'])
        self.cache['d_gamma_cache'] = self.cache['d_gamma']
        self.cache['d_beta_cache'] = self.cache['d_beta']

    

class Linear():
    def __init__(self, in_dim, out_dim, bias=True):
        self.has_bias = False
        if bias:
            self.has_bias = True
            self.bias = np.zeros((out_dim, 1))
        
        self.cache = {
                      'w_grad_cache' : np.zeros((out_dim, in_dim)),
                      'b_grad_cache' : np.zeros((out_dim, 1))
                     }
        self.weight = np.random.randn(out_dim, in_dim) * np.sqrt(1./in_dim)
        
    def forward(self, x):
        self.cache['x'] = x
        
        if self.has_bias:
            self.cache['z'] = self.weight @ x + self.bias
        else:
            self.cache['z'] = self.weight @ x
    
        return self.cache['z']

    def backward(self, dz):
        batch_size = dz.shape[0]
        self.cache['w_grad'] = (1./batch_size) * dz @ self.cache['x'].T
        if self.has_bias:
            self.cache['b_grad'] = (1./batch_size) * np.sum(dz, axis=1, keepdims=True)
            
        return self.weight.T @ dz
        
        
    
    def update(self, learning_rate, Beta=0.95):
        if self.has_bias:
            self.bias = self.bias - learning_rate * (Beta*self.cache['b_grad_cache'] + (1-Beta) * self.cache['b_grad'])
            self.cache['b_grad_cache'] = self.cache['b_grad']
            
        self.weight = self.weight - learning_rate * (Beta*self.cache['w_grad_cache'] + (1-Beta) * self.cache['w_grad'])
        self.cache['w_grad_cache'] = self.cache['w_grad']

