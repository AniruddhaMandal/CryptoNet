from NeuralNet.neurla_network import NeuralNetwork
import numpy as np

layers = [784, 30, 10]

test_network = NeuralNetwork(layers)
Data = np.load('data/mnist_data/Data.npy')
Labels = np.load('data/mnist_data/Labels.npy')
X = Data[1]
y = Labels[1]

def test_feedforward():
    test_network = NeuralNetwork(layers)
    output = test_network.feedforward(X)
    assert output.shape == (10,)

def test_backpropagation():
    grad_weights, grad_biases = test_network.backpropagation(X, y)

def test_SGD():
    test_network.SGD(Data, Labels)

# def test_accuracy():
#     acc = test_network.Accuracy(Data, Labels)
#     print("accuray", acc)