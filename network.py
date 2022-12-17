from util.activation import *

import numpy as np


# class NN():
#     def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
#         self.input_data = np.random.randn(1, input_size)
#         self.w1 = np.random.randn(input_size, hidden_1_size)
#         self.b1 = np.zeros((1, hidden_1_size))
        
#         self.w2 = np.random.randn(hidden_1_size, hidden_2_size)
#         self.b2 = np.zeros((1, hidden_2_size)) 

#         self.w3 = np.random.randn(hidden_2_size, output_size)
#         self.b3 = np.zeros((1, output_size))
        
#     # Loss Functions
#     def cross_entropy(self, Y, Y_prediction):
#         return -np.sum(np.multiply(Y, np.log(Y_prediction + 1e-07)))

#     def Sigmoid(self, z):
#         return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-7))

#     def Softmax(self, z):
#         z = z - np.max(z)
#         return np.exp(z)/np.sum(np.exp(z))

#     def Relu(self, z):
#         return np.maximum(z, 0)

#     def forward(self, input_data):
#         self.input_data = input_data
#         self.h1_out = Relu(input_data.dot(self.w1) + self.b1)
#         self.h2_out = Relu(self.h1_out.dot(self.w2) + self.b2)
#         self.output_layer = Softmax(self.h2_out.dot(self.w3) + self.b3)
#         return self.output_layer

#     # Backward Propagation
#     def backward(self, target):

#         # corss_entropy loss derivative
#         Loss_to_z_grad = (self.output_layer - target) 

#         self.b3_grad = Loss_to_z_grad
#         self.w3_grad = self.h2_out.T.dot(Loss_to_z_grad)

#         Activation_2_grad = Loss_to_z_grad.dot(self.w3.T)
#         Activation_2_grad[self.h2_out<0] = 0


#         self.b2_grad = Activation_2_grad
#         self.w2_grad = self.h1_out.T.dot(Activation_2_grad)

        
#         Activation_1_grad = Activation_2_grad.dot(self.w2.T)
#         Activation_1_grad[self.h1_out<0] = 0     

#         self.b1_grad = Activation_1_grad
#         self.w1_grad = self.input_data.T.dot(Activation_1_grad)


#     # Update Weights
#     def update(self, learning_rate=1e-03):
#         self.w1 = self.w1 - learning_rate * self.w1_grad
#         self.b1 = self.b1 - learning_rate * self.b1_grad

#         self.w2 = self.w2 - learning_rate * self.w2_grad
#         self.b2 = self.b2 - learning_rate * self.b2_grad

#         self.w3 = self.w3 - learning_rate * self.w3_grad
#         self.b3 = self.b3 - learning_rate * self.b3_grad

class NN():
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        self.input_data = np.random.randn(input_size, 1)
        self.w1 = np.random.randn(hidden_1_size, input_size)
        self.b1 = np.zeros((hidden_1_size, 1))
        
        self.w2 = np.random.randn(hidden_2_size, hidden_1_size)
        self.b2 = np.zeros((hidden_2_size, 1)) 

        self.w3 = np.random.randn(output_size, hidden_2_size)
        self.b3 = np.zeros((output_size, 1))

    def acc_test(self, input):
        tmp_h1 = Relu((self.w1).dot(input) + self.b1)
        tmp_h2 = Relu(self.w2.dot(tmp_h1) + self.b2)
        tmp_out = Softmax(self.w3.dot(tmp_h2) + self.b3)
        return tmp_out

    # Feed Placeholder

    def forward(self, input_data):

        self.input_data = input_data
        self.h1_out = Relu(self.w1 @ self.input_data + self.b1)
        self.h2_out = Relu(self.w2 @ self.h1_out + self.b2)
        self.output_layer = Softmax(self.w3 @ self.h2_out + self.b3)
        return self.output_layer

    def backward(self, target):

        # corss_entropy loss derivative
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
        self.w1_grad = Activation_1_grad @ self.input_data.T


    # Update Weights
    def update(self, learning_rate=1e-03):
        self.w1 = self.w1 - learning_rate * self.w1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad

        self.w2 = self.w2 - learning_rate * self.w2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad

        self.w3 = self.w3 - learning_rate * self.w3_grad
        self.b3 = self.b3 - learning_rate * self.b3_grad