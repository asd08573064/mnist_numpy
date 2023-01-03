from util.activation import *
import numpy as np

class NN():
    def __init__(self, input_size=784,hidden_0_size=256, output_size=10, learning_rate=0.1):
        self.input_data = np.random.randn(input_size, 1)
        self.learning_rate = learning_rate
        
        self.w0 = np.random.randn(hidden_0_size, input_size) * np.sqrt(1./input_size)
        self.b0 = np.zeros((hidden_0_size, 1))
        
        self.w1 = np.random.randn(output_size, hidden_0_size) * np.sqrt(1./hidden_0_size)
        self.b1 = np.zeros((output_size, 1))
        
        self.w0_grad_cache = np.zeros((hidden_0_size, input_size)) 
        self.b0_grad_cache = np.zeros((hidden_0_size, 1))  
        
        self.w1_grad_cache = np.zeros((output_size, hidden_0_size)) 
        self.b1_grad_cache = np.zeros((output_size, 1))  
        
    def forward(self, input_data):
        self.input_data = input_data
        self.z0 = self.w0 @ self.input_data.T + self.b0
        self.h0_out = Relu(self.z0)
        self.output_layer = Softmax(self.w1 @ self.h0_out + self.b1)
        return self.output_layer

    def backward(self, target):
        batch_size = target.shape[0]
        Loss_to_z_grad = self.output_layer - target.T

        self.b1_grad = (1./batch_size) * np.sum(Loss_to_z_grad, axis=1, keepdims=True)
        self.w1_grad = (1./batch_size) * Loss_to_z_grad @ self.h0_out.T

        Activation_1_grad = self.w1.T @ Loss_to_z_grad
        Activation_1_grad[self.z0<0] = 0   

        self.b0_grad = (1./batch_size) * np.sum(Activation_1_grad, axis=1, keepdims=True)
        self.w0_grad = (1./batch_size) * Activation_1_grad @ self.input_data

    # Update Weights
    def update(self, Beta=0.95):

        self.w0 = self.w0 - self.learning_rate * (Beta*self.w0_grad_cache + (1-Beta) * self.w0_grad)
        self.b0 = self.b0 - self.learning_rate * (Beta*self.b0_grad_cache + (1-Beta) * self.b0_grad)
        
        self.w0_grad_cache = self.w0_grad
        self.b0_grad_cache = self.b0_grad
        
        self.w1 = self.w1 - self.learning_rate * (Beta*self.w1_grad_cache + (1-Beta) * self.w1_grad)
        self.b1 = self.b1 - self.learning_rate * (Beta*self.b1_grad_cache + (1-Beta) * self.b1_grad)
        
        self.w1_grad_cache = self.w1_grad
        self.b1_grad_cache = self.b1_grad

    def schduler_step(self, decay=0.1, epoch=0):
        self.learning_rate *= (1. /(1. + decay * (epoch)))