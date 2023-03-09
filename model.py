from util.layers import Linear, Batch_Norm, Softmax, Sigmoid, Cross_Entropy
import numpy as np

class NN():
    def __init__(self, input_size=784, hidden_0_size=256, hidden_1_size=128, output_size=10, learning_rate=0.5):
        
        self.learning_rate = learning_rate
        self.bn0 = Batch_Norm()
        self.bn1 = Batch_Norm()
        self.layer0 = Linear(input_size, hidden_0_size)
        self.layer1 = Linear(hidden_0_size, hidden_1_size)
        self.layer2 = Linear(hidden_1_size, output_size)
        self.softmax = Softmax()
        self.sigmoid0 = Sigmoid()
        self.sigmoid1 = Sigmoid()
        
    def forward(self, input_data):
        """forward model

        Args:
            input_data (numpy.darray): input feature with dimension [feature_dim, batch_size]

        Returns:
            numpy.darray: output predicted logits
        """
        self.z0 = self.layer0.forward(input_data)
        self.x0 = self.bn0.forward(self.z0)
        self.A0 = self.sigmoid0.forward(self.x0)
        
        self.z1 = self.layer1.forward(self.A0)
        self.x1 = self.bn1.forward(self.z1)
        self.A1 = self.sigmoid1.forward(self.x1)
        
        self.output_layer = self.softmax.forward(self.layer2.forward(self.A1))
        
        return self.output_layer

    def backward(self, d_C):
        """_summary_

        Args:
            d_C (numpy.darray): gradient from the loss function.
        """
        
        dz2 = self.softmax.backward(d_C)
        dA1 = self.layer2.backward(dz2)
        dx1 = self.sigmoid1.backward(dA1) 
        dz1 = self.bn1.backward(dx1)
        dA0 = self.layer1.backward(dz1)
        dx0 = self.sigmoid0.backward(dA0) 
        dz0 = self.bn0.backward(dx0)
        ___ = self.layer0.backward(dz0)

    # Update Weights
    def update(self, Beta=0.96):
        self.layer0.update(self.learning_rate, Beta=Beta)
        self.layer1.update(self.learning_rate, Beta=Beta)
        self.layer2.update(self.learning_rate, Beta=Beta)
        self.bn0.update(self.learning_rate, Beta=Beta)
        self.bn1.update(self.learning_rate, Beta=Beta)
        
    def schduler_step(self, decay=0., epoch=0):
        self.learning_rate *= (1. /(1. + decay * (epoch)))