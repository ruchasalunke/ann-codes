import numpy as np

# Input (X: hours sleeping, hours studying), Output (y: test scores)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100

class NeuralNetwork:
    def _init_(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (2x3)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1)
        
    def sigmoid(self, s, deriv=False):
        if deriv:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))
    
    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.sigmoid(self.z3)
        return output
    
    def backward(self, X, y, output, learning_rate=0.1):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)
        
        self.W1 += learning_rate * X.T.dot(self.z2_delta)
        self.W2 += learning_rate * self.z2.T.dot(self.output_delta)
    
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)

NN = NeuralNetwork()

for i in range(1000):
    if i % 100 == 0:
        print(f"Loss after {i} iterations: {np.mean(np.square(y - NN.feedForward(X)))}")
    NN.train(X, y)

print("\nInput:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", NN.feedForward(X))
print("Final Loss:", np.mean(np.square(y - NN.feedForward(X))))
