from util.activation import *
from util.layers import Linear
import numpy as np

class NN():
    def __init__(self, input_size=784, hidden_0_size=256, hidden_1_size=128, output_size=10, learning_rate=0.8):
        self.input_data = np.random.randn(input_size, 1)
        self.learning_rate = learning_rate
        self.layer0 = Linear(input_size, hidden_0_size)
        self.layer1 = Linear(hidden_0_size, hidden_1_size)
        self.layer2 = Linear(hidden_1_size, output_size)
        
    def forward(self, input_data):
        self.input_data = input_data
        self.A0 = Sigmoid(self.layer0.forward(input_data.T))
        self.A1 = Sigmoid(self.layer1.forward(self.A0))
        self.output_layer = Softmax(self.layer2.forward(self.A1))
        return self.output_layer

    def backward(self, target):
        C_dA = Cross_Entropy(self.output_layer, target.T, derivative=True)
        A_dz = Softmax(self.output_layer, derivative=True)
        C_dz = C_dA * A_dz

        A1_grad = self.layer2.weight.T @ C_dz
        dz1 = A1_grad * Sigmoid(self.layer1.cache['z'], derivative=True) 
        
        A0_grad = self.layer1.weight.T @ dz1
        dz0 = A0_grad * Sigmoid(self.layer0.cache['z'], derivative=True) 
        
        self.layer2.backward(C_dz,  self.A1.T)
        self.layer1.backward(dz1,  self.A0.T)
        self.layer0.backward(dz0,  self.input_data)

    # Update Weights
    def update(self, Beta=0.96):
        self.layer0.update(self.learning_rate, Beta=Beta)
        self.layer1.update(self.learning_rate, Beta=Beta)
        self.layer2.update(self.learning_rate, Beta=Beta)
        
    def schduler_step(self, decay=0., epoch=0):
        self.learning_rate *= (1. /(1. + decay * (epoch)))