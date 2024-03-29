from util.layers import Linear, Batch_Norm, Softmax, Sigmoid, ReLU

class NN():
    def __init__(self, input_size=784, hidden_0_size=256, hidden_1_size=128, output_size=10, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.module_list = [
            Linear(input_size, hidden_0_size),
            Batch_Norm(),
            Sigmoid(),
            Linear(hidden_0_size, hidden_1_size),
            Batch_Norm(),
            Sigmoid(),
            Linear(hidden_1_size, output_size),
            Softmax()
        ]
        
    def forward(self, x):
        """forward pass

        Args:
            input_data (numpy.darray): input feature with dimension [feature_dim, batch_size]

        Returns:
            numpy.darray: output predicted logits
        """
        for layer in self.module_list:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """backward pass

        Args:
            d_C (numpy.darray): gradient from the loss function.
        """
        for layer in reversed(self.module_list):
            grad = layer.backward(grad)

    def update(self, Beta=0.96):
        """Update the parameters

        Args:
            Beta (float, optional): Momemtum for SGD-M algorithm. Defaults to 0.96.
        """
        for layer in self.module_list:
            if "cache" in layer.__dict__: # check whether the layer has 
                layer.update(self.learning_rate, Beta=Beta)
        
    def schduler_step(self, decay=0., epoch=0):
        self.learning_rate *= (1. /(1. + decay * (epoch)))