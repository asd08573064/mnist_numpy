from util.activation import *
import numpy as np

class NN():
    def __init__(self, input_size=784,hidden_0_size=512, hidden_1_size=256, hidden_2_size=128, output_size=10, learning_rate=1e-5):
        self.input_data = np.random.randn(input_size, 1)
        self.learning_rate = learning_rate
        
        self.w0 = np.random.randn(hidden_0_size, input_size)
        self.b0 = np.zeros((hidden_0_size, 1))
        
        self.w1 = np.random.randn(hidden_1_size, hidden_0_size)
        self.b1 = np.zeros((hidden_1_size, 1))
        
        self.w2 = np.random.randn(hidden_2_size, hidden_1_size)
        self.b2 = np.zeros((hidden_2_size, 1)) 

        self.w3 = np.random.randn(output_size, hidden_2_size)
        self.b3 = np.zeros((output_size, 1))
        
        self.w0_grad_cache = np.zeros((hidden_0_size, input_size)) 
        self.b0_grad_cache = np.zeros((hidden_0_size, 1))  
        
        self.w1_grad_cache = np.zeros((hidden_1_size, hidden_0_size)) 
        self.b1_grad_cache = np.zeros((hidden_1_size, 1))  

        self.w2_grad_cache = np.zeros((hidden_2_size, hidden_1_size)) 
        self.b2_grad_cache = np.zeros((hidden_2_size, 1))  
        
        self.w3_grad_cache = np.zeros((output_size, hidden_2_size)) 
        self.b3_grad_cache = np.zeros((output_size, 1))  

    def forward(self, input_data):
        self.input_data = input_data
        self.h0_out = Relu(self.w0 @ self.input_data + self.b0)
        self.h1_out = Relu(self.w1 @ self.h0_out + self.b1)
        self.h2_out = Relu(self.w2 @ self.h1_out + self.b2)
        self.output_layer = Softmax(self.w3 @ self.h2_out + self.b3)
        return self.output_layer

    def backward(self, target):
        Loss_to_z_grad = (self.output_layer - target) 

        self.b3_grad = Loss_to_z_grad
        self.w3_grad = Loss_to_z_grad @ self.h2_out.T
        
        Activation_2_grad = self.w3.T @ Loss_to_z_grad
        Activation_2_grad[self.h2_out<0] = 0

        self.b2_grad = Activation_2_grad
        self.w2_grad = Activation_2_grad @ self.h1_out.T

        Activation_1_grad = self.w2.T @ Activation_2_grad
        Activation_1_grad[self.h1_out<0] = 0     

        self.b1_grad = Activation_1_grad
        self.w1_grad = Activation_1_grad @ self.h0_out.T
        
        Activation_0_grad = self.w1.T @ Activation_1_grad
        Activation_0_grad[self.h0_out<0] = 0     

        self.b0_grad = Activation_0_grad
        self.w0_grad = Activation_0_grad @ self.input_data.T

    # Update Weights
    def update(self, Beta=0.1, epoch=0, decay=0.6):
        self.learning_rate *= (1. /(1. + decay * epoch))
        
        self.w0 = self.w0 - self.learning_rate * (Beta*self.w0_grad_cache + (1-Beta) * self.w0_grad)
        self.b0 = self.b0 - self.learning_rate * (Beta*self.b0_grad_cache + (1-Beta) * self.b0_grad)
        
        self.w3_grad_cache = self.w3_grad
        self.b3_grad_cache = self.b3_grad
        
        self.w1 = self.w1 - self.learning_rate * (Beta*self.w1_grad_cache + (1-Beta) * self.w1_grad)
        self.b1 = self.b1 - self.learning_rate * (Beta*self.b1_grad_cache + (1-Beta) * self.b1_grad)
        
        self.w1_grad_cache = self.w1_grad
        self.b1_grad_cache = self.b1_grad

        self.w2 = self.w2 - self.learning_rate * (Beta*self.w2_grad_cache + (1-Beta) * self.w2_grad)
        self.b2 = self.b2 - self.learning_rate * (Beta*self.b2_grad_cache + (1-Beta) * self.b2_grad)
        
        self.w2_grad_cache = self.w2_grad
        self.b2_grad_cache = self.b2_grad

        self.w3 = self.w3 - self.learning_rate * (Beta*self.w3_grad_cache + (1-Beta) * self.w3_grad)
        self.b3 = self.b3 - self.learning_rate * (Beta*self.b3_grad_cache + (1-Beta) * self.b3_grad)
        
        self.w3_grad_cache = self.w3_grad
        self.b3_grad_cache = self.b3_grad