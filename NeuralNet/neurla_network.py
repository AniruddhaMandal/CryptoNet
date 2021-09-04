import numpy as np
import logging
_log = logging.getLogger()

class NeuralNetwork():
    def __init__(self, layers:list) -> None:
        self.layers  = layers
        self.weights = np.array([np.random.random((layer_after, layer_before+1)) for layer_before, layer_after in zip(layers[:-1],layers[1:])],dtype=object)

    def feedforward(self, x:np.ndarray) -> np.ndarray:
        for weight in self.weights:
            x = np.append(x, 1)
            x = np.dot(weight, x)
            x = self.sigmoid(x)
        return x

    def reverse_auto_dev(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        _input  = x
        inputs  = np.zeros(len(self.weights), dtype=object) # a
        outputs = np.zeros(len(self.weights), dtype=object) # z
        for i, weight in enumerate(self.weights):
            _input       = np.append(_input,1)
            inputs[i]    = _input
            #_log.debug(f"Weight shape: {weight.shape}, _input shape : {_input.shape}")
            #_log.debug(f"Weight:{weight},\n input:{_input}")
            output       = np.dot(weight, _input)
            #_log.debug(f"Product: {output}")
            outputs[i]   = output
            _input       = self.sigmoid(output)
            grad_weights = np.zeros(len(self.weights), dtype=object)
        grad_z = np.dot((_input-y), self.sigmoid_prime(outputs[-1]))
        #_log.debug(outputs[-1])
        #_log.debug(f"Outputs: {outputs}")
        for l in range(len(self.layers)):
            grad_weight = np.dot(grad_z, inputs[-l-1])
            grad_weights[-l-1] = grad_weight
            #_log.debug(f"Second to last output: {outputs[-l-2]}")
            _log.debug(f"Sigmoid Prime: {self.sigmoid_prime(outputs[-l-2]).shape}")
            _log.debug(f"Input Shape: {inputs[-l-2]}")
            #_log.debug(f"Last Weight: {self.weights[-l-1].T}")
            var = np.multiply(self.sigmoid_prime(outputs[-l-2])[:,np.newaxis],self.weights[-l-1].T)
            grad_z = np.dot(var, grad_z)
        return grad_weights

    def accuracy(self,x,y):
        return np.sum([self.predict(i)==j for i,j in zip(x,y)])/len(x)

    def cumulative_gradient(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        grad_weights = np.array([np.zeros((i,j+1)) for i,j in zip(self.layers[1:],self.layers[:-1])])	
        for data_x, data_y in zip(x,y):
            print("#", end="")
            grad_weights = grad_weights + self.reverse_auto_dev(data_x, data_y)
        print("\n")
        return grad_weights

    def stochastic_gradient_descent(self, x:np.ndarray,y:np.ndarray, batch_size:int, learning_rate:float) -> None:
        permutation  = np.random.permutation(len(x))
        x = x[permutation]
        y = y[permutation]
        batches_x = [x[i:i+batch_size] for i in range(0,len(x), batch_size)]
        batches_y = [y[i:i+batch_size] for i in range(0,len(y), batch_size)]
        i = 0
        for batch_x, batch_y in zip(batches_x, batches_y):
            print(f"[{i}] Training for batch no {i}")
            self.weights = self.weights - learning_rate*self.cumulative_gradient(batch_x, batch_y)

            print(f"[{i}] Trained for batch no {i}: {self.accuracy(batch_x, batch_y)}")

    def train(self, data_x:np.ndarray, data_y:np.ndarray, batch_size:int=1000,learning_rate:float=0.01) -> None:
        self.stochastic_gradient_descent(data_x,data_y,batch_size=batch_size,learning_rate=learning_rate)

    def predict_probability(self, dataset:np.ndarray) -> np.ndarray:
        proba = [self.feedforward(data) for data in dataset]
        return proba

    def predict(self, dataset:np.ndarray) -> np.ndarray:
        proba = self.predict_probability(dataset)
        output = [np.max(arr) for arr in proba]
        return output

    def sigmoid(self, x:np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(x))
    
    def sigmoid_prime(self, x:np.ndarray) -> np.ndarray:
        sig = self.sigmoid(x)
        return sig*(1-sig)

    