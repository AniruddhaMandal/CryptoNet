from data import mnist_loader
data, data_labels = mnist_loader.get_data("data/mnist_data/train-images-idx3-ubyte.gz","data/mnist_data/train-labels-idx1-ubyte.gz")

def test_data_labels_length():
    assert len(data) == len(data_labels)

def test_data_dimension():
    print(data.shape)
    assert data.shape == (len(data_labels), 784)