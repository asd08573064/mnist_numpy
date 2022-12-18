from util.activation import *
import numpy as np

class NN():
    def __init__(self, input_size=784,hidden_0_size=512, hidden_1_size=256, hidden_2_size=128, output_size=10, learning_rate=1e-5):
        self.learning_rate = learning_rate
        self.paras = {
            'layer0' : { # dummy layer for input data
                'out' : np.random.randn(input_size, 1) 
            },
            'layer1' : {
                'w' : np.random.randn(hidden_0_size, input_size),
                'b' : np.zeros((hidden_0_size, 1)),
                'w_grad_cache' : np.zeros((hidden_0_size, input_size)),
                'b_grad_cache' : np.zeros((hidden_0_size, 1)),
                'out' : np.zeros((hidden_0_size, 1))
            },
            'layer2' : {
                'w' : np.random.randn(hidden_1_size, hidden_0_size),
                'b' : np.zeros((hidden_1_size, 1)),
                'w_grad_cache' : np.zeros((hidden_1_size, hidden_0_size)),
                'b_grad_cache' : np.zeros((hidden_1_size, 1)),
                'out' : np.zeros((hidden_1_size, 1))
            },
            'layer3' : {
                'w' : np.random.randn(hidden_2_size, hidden_1_size), 
                'b' : np.zeros((hidden_2_size, 1)),
                'w_grad_cache' : np.zeros((hidden_2_size, hidden_1_size)),
                'b_grad_cache' : np.zeros((hidden_2_size, 1)),
                'out' : np.zeros((hidden_2_size, 1))
            },
            'layer4' : {
                'w' : np.random.randn(output_size, hidden_2_size),
                'b' : np.zeros((output_size, 1)),
                'w_grad_cache' : np.zeros((output_size, hidden_2_size)),
                'b_grad_cache' : np.zeros((output_size, 1)),
                'out' : np.zeros((output_size, 1))
            }
        }

    def forward(self, input_data):
        self.paras['layer0']['out'] = input_data
        self.paras['layer1']['out'] = Relu(self.paras['layer1']['w'] @ input_data + self.paras['layer1']['b'])
        self.paras['layer2']['out'] = Relu(self.paras['layer2']['w'] @ self.paras['layer1']['out'] + self.paras['layer2']['b'])
        self.paras['layer3']['out'] = Relu(self.paras['layer3']['w'] @ self.paras['layer2']['out'] + self.paras['layer3']['b'])
        self.paras['layer4']['out'] = Softmax(self.paras['layer4']['w'] @ self.paras['layer3']['out'] + self.paras['layer4']['b'])
        return self.paras['layer4']['out']

    def backward(self, target):
        for layer_id in range(len(self.paras)-1, 0, -1):
            next_layer = 'layer{}'.format(str(layer_id+1))
            layer = 'layer{}'.format(str(layer_id))
            pre_layer = 'layer{}'.format(str(layer_id-1))
            if layer_id == 4:
                Activation_grad = (self.paras[layer]['out'] - target) 
            else:
                Activation_grad = self.paras[next_layer]['w'].T @ Activation_grad
                Activation_grad[self.paras[layer]['out'] < 0] = 0
            
            self.paras[layer]['b_grad'] = Activation_grad
            self.paras[layer]['w_grad'] = Activation_grad @ self.paras[pre_layer]['out'].T

    # Update Weights
    def update(self, Beta=0.1, epoch=0, decay=0.6):
        self.learning_rate *= (1. /(1. + decay * epoch))
        for layer_id in range(1, len(self.paras)):
            layer = 'layer{}'.format(str(layer_id))
            self.paras[layer]['w'] = self.paras[layer]['w'] - self.learning_rate * (Beta*self.paras[layer]['w_grad'] + (1-Beta) * self.paras[layer]['w_grad_cache'])
            self.paras[layer]['b'] = self.paras[layer]['b'] - self.learning_rate * (Beta*self.paras[layer]['b_grad'] + (1-Beta) * self.paras[layer]['b_grad_cache'])
            self.paras[layer]['w_grad'] = self.paras[layer]['w_grad']
            self.paras[layer]['b_grad'] = self.paras[layer]['b_grad']