import numpy as np

class Batch_Norm():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gamma = 0
        self.beta = 0
        self.cache = {}
        
    def forward(self, x):
        epsilon = 1e-9
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.x_hat = (x - self.mu) / (np.sqrt(self.var) + epsilon)
        self.y = self.gamma * self.x_hat + self.beta
        return self.y
    
    def backward(self, da):
        self.cache['d_beta'] = np.sun(da, axis=0)
        self.cache['d_gamma'] = np.sun(da * self.x_hat, axis=0)
        self.cache['d_x'] = 0
        # TBD

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

    def backward(self, da, a_pre):
        batch_size = da.shape[0]
        self.cache['w_grad'] = (1./batch_size) * da @ a_pre
        if self.has_bias:
            self.cache['b_grad'] = (1./batch_size) * np.sum(da, axis=1, keepdims=True)
    
    def update(self, learning_rate, Beta=0.95):
        if self.has_bias:
            self.bias = self.bias - learning_rate * (Beta*self.cache['b_grad_cache'] + (1-Beta) * self.cache['b_grad'])
            self.cache['b_grad_cache'] = self.cache['b_grad']
            
        self.weight = self.weight - learning_rate * (Beta*self.cache['w_grad_cache'] + (1-Beta) * self.cache['w_grad'])
        self.cache['w_grad_cache'] = self.cache['w_grad']
        
    