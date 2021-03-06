# -*- coding: utf-8 -*-
"""mnist_self_implement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ge-Kv__rNc5lF1ak8M0EytyMv6KbaEyH
"""

from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255.).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

import numpy as np


class NN():
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        self.input_data = np.random.randn(1, input_size)
        self.w1 = np.random.randn(input_size, hidden_1_size)
        self.b1 = np.zeros((1, hidden_1_size))
        
        self.w2 = np.random.randn(hidden_1_size, hidden_2_size)
        self.b2 = np.zeros((1, hidden_2_size)) 

        self.w3 = np.random.randn(hidden_2_size, output_size)
        self.b3 = np.zeros((1, output_size))


    def Sigmoid(self, z):
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-7))

    def Softmax(self, z):
        z = z - np.max(z)
        return np.exp(z)/np.sum(np.exp(z))

    def Relu(self, z):
        return np.maximum(z, 0)

    def acc_test(self, input):
        tmp_h1 = self.Relu(input.dot(self.w1) + self.b1)
        tmp_h2 = self.Relu(tmp_h1.dot(self.w2) + self.b2)
        tmp_out = self.Softmax(tmp_h2.dot(self.w3) + self.b3)
        return tmp_out

    # Feed Placeholder

    def forward(self, input_data):

        self.input_data = input_data
        self.h1_out = self.Relu(input_data.dot(self.w1) + self.b1)
        self.h2_out = self.Relu(self.h1_out.dot(self.w2) + self.b2)
        self.output_layer = self.Softmax(self.h2_out.dot(self.w3) + self.b3)

    # Backward Propagation

    def backward(self, target):

        # corss_entropy loss derivative
        Loss_to_z_grad = (self.output_layer - target) # correct 

        self.b3_grad = Loss_to_z_grad
        self.w3_grad = self.h2_out.T.dot(Loss_to_z_grad) # correct

        Activation_2＿grad = Loss_to_z_grad.dot(self.w3.T) # correct
        Activation_2_grad[self.h2_out<0] = 0


        self.b2_grad = Activation_2＿grad
        self.w2_grad = self.h1_out.T.dot(Activation_2＿grad)

        
        Activation_1＿grad = Activation_2＿grad.dot(self.w2.T)
        Activation_1_grad[self.h1_out<0] = 0     

        self.b1_grad = Activation_1＿grad
        self.w1_grad = self.input_data.T.dot(Activation_1＿grad)


    # Update Weights
    def update(self, learning_rate=1e-03):
        self.w1 = self.w1 - learning_rate * self.w1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad

        self.w2 = self.w2 - learning_rate * self.w2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad

        self.w3 = self.w3 - learning_rate * self.w3_grad
        self.b3 = self.b3 - learning_rate * self.b3_grad

    # Loss Functions
    def cross_entropy(self, Y, Y_prediction):
        return -np.sum(np.multiply(Y, np.log(Y_prediction + 1e-07)))

    def print_accuracy(self):
        correct = 0
        loss = 0
        for i in range(y_train.shape[0]):
            index = self.acc_test(x_train[i])
            one_hot = 0
            for check in range(y_train[i].shape[0]):
                pre = np.argmax(index)
                if y_train[i][check] == 1:
                    one_hot = check
                    break
                loss += self.cross_entropy(index, y_train[i])
            if pre == one_hot:
                correct += 1
        print('error = ', loss/y_train.shape[0])
        print('accuracy = ', correct/y_train.shape[0])



import random
 mnist_nn = NN(input_size = 784, hidden_1_size = 700, hidden_2_size = 700,output_size = 10)


 
for i in range(20):
    for j in range(1000):
        index = random.randint(0,x_train.shape[0])-1
        mnist_nn.forward(x_train[[index]])
        mnist_nn.backward(y_train[index])
        mnist_nn.update()
    print(i)
    mnist_nn.print_accuracy()