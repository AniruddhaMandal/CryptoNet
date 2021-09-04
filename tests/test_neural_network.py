from NeuralNet import neurla_network
from data import mnist_loader
layers = [784, 30, 10]
test_network = neurla_network.NeuralNetwork(layers)

def test_weights_dimension():
    assert type(2) == int
    for i,weight in enumerate(test_network.weights):
            assert weight.shape == (layers[i+1], layers[i]+1)

def test_neural_network():
    X, y = mnist_loader.get_data("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    test_network.train(X, y)         