import numpy as np

class NeuralNetwork():
    def __init__(self, layers: list):
        self.layers = layers
        self.count_layer = len(layers)
        self.weights = np.array([np.random.random((i,j)) for i,j in zip(layers[1:], layers[:-1])],dtype=object)
        self.biases = np.array([np.random.random(i) for i in layers[1:]], dtype=object)

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        a_input = X 
        for weight, bias in zip(self.weights, self.biases):
            X = np.dot(weight, X) + bias
            X = self.sigmoid(X)
        return X

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-Z))

    def sigmoid_prime(self, Z: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(Z)
        return sig*(1-sig)

    def backpropagation(self, X: np.ndarray, y: np.ndarray):
        #Z = WxA + B; A' = sigmoid(Z) where A' is the input for next layer or the output of the final layer;
        inputs = np.array([np.zeros(layer) for layer in self.layers[:-1]], dtype=object) #a
        activations = np.array([np.zeros(layer) for layer in self.layers[1:]], dtype=object)
        for weight, bias, i in zip(self.weights, self.biases, range(self.count_layer-1)):
            inputs[i] = X
            activation = np.dot(weight,X) + bias
            activations[i] = activation 
            X = self.sigmoid(activation)
        # At this point X is the infeted value i.e. Y.
        grad_weights = np.array([np.zeros(weight.shape) for weight in self.weights], dtype=object)
        grad_biases = np.array([np.zeros(bias.shape) for bias in self.biases], dtype=object)
        grad_input = (X-y)
        for l in range(self.count_layer-1):
            grad_activation = np.multiply(grad_input, self.sigmoid_prime(activations[-l-1]))
            grad_weights[-l-1] = np.dot(grad_activation[:, np.newaxis], inputs[-l-1][:,np.newaxis].T)
            grad_bias = grad_activation
            grad_biases[-l-1] = grad_bias 
            grad_input = np.dot(self.weights[-l-1].T, grad_activation)
        return grad_weights, grad_biases

    def SGD(self, Dataset: np.ndarray, Labels: np.ndarray, learning_rate: np.float64 = 0.01, batch_size: np.int64 = 1000) -> None:
        rand_state = np.random.get_state()
        np.random.shuffle(Dataset)
        np.random.set_state(rand_state)
        np.random.shuffle(Labels)
        counter = 1
        batched_data = [Dataset[i:i+batch_size] for i in range(0,len(Dataset), batch_size)]
        batched_labels = [Labels[i:i+batch_size] for i in range(0, len(Labels),batch_size)]
        for data, labels in zip(batched_data,batched_labels):
            batch_grad_weights = np.array([np.zeros(weight.shape) for weight in self.weights], dtype=object)
            batch_grad_biases = np.array([np.zeros(bias.shape) for bias in self.biases], dtype=object)
            print(f"Training Batch [{counter}]: ")
            for X, y in zip(data,labels):
                grad_weights, grad_biases = self.backpropagation(X, y)
                batch_grad_weights = batch_grad_weights + grad_weights
                batch_grad_biases = batch_grad_biases + grad_biases 
                print("#", end="")
            self.weights = self.weights - learning_rate*batch_grad_weights
            self.biases = self.biases - learning_rate*batch_grad_biases
            print(f"[{counter}]Accuracy: {self.Accuracy(data,labels)}%")
            counter = counter + 1

    def train(self, Dataset: np.ndarray, Labels: np.ndarray, learning_rate: np.float64 = 0.01, batch_size: np.int64 = 1000, cycles: np.int16 = 1) -> None:
        for i in range(cycles):
            self.SGD(Dataset, Labels, learning_rate, batch_size)

    def predict_proba(self, Dataset: np.ndarray) -> np.ndarray:
        results = [self.feedforward(X) for X in Dataset]
        return results
    
    def predict_single(self, X: np.ndarray) -> np.int64:
        return self.feedforward(X).argmax()

    def predict(self, Dataset: np.ndarray) -> np.ndarray:
        result = [self.predict_single(X) for X in Dataset]
        return result 
    
    def Error_rate(self, Dataset, Labels):
        result = (self.predict(Dataset)-Labels).sum()
        result = result/len(Dataset)
        return result
    
    def Accuracy(self, Dataset, Labels):
        Labels = np.array([label.argmax() for label in Labels], dtype=object)
        result = np.array([p == l for p, l in zip(self.predict(Dataset), Labels)], dtype=object)
        result = result.sum()/len(Dataset)
        return result*100
