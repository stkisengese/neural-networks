import numpy as np

class Neuron:
    def __init__(self, weight1, weight2, bias, regression=False):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias
        self.regression = regression
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def identity(self, x):
        """Identity activation function for regression"""
        return x
    
    def feedforward(self, x1, x2):
        # Weighted sum + bias
        total = (x1 * self.weights_1) + (x2 * self.weights_2) + self.bias
        
        # Choose activation function
        if self.regression:
            return self.identity(total)
        else:
            return self.sigmoid(total)

class OurNeuralNetwork:
    def __init__(self, neuron_h1, neuron_h2, neuron_o1):
        self.h1 = neuron_h1
        self.h2 = neuron_h2
        self.o1 = neuron_o1
    
    def feedforward(self, x1, x2):
        out_h1 = self.h1.feedforward(x1, x2)
        out_h2 = self.h2.feedforward(x1, x2)
        y = self.o1.feedforward(out_h1, out_h2)
        return y