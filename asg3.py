import numpy as np
class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x): return x * (1 - x)

    def train(self, X, y, lr=0.1, epochs=10000):
        for _ in range(epochs):
            h_in = np.dot(X, self.w1) + self.b1
            h_out = self.sigmoid(h_in)
            o_in = np.dot(h_out, self.w2) + self.b2
            output = self.sigmoid(o_in)

            error = y - output
            d_output = error * self.sigmoid_deriv(output)
            d_hidden = d_output.dot(self.w2.T) * self.sigmoid_deriv(h_out)

            self.w2 += h_out.T.dot(d_output) * lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
            self.w1 += X.T.dot(d_hidden) * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
        return output

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
nn = FeedForwardNN(2, 2, 1)
output = nn.train(X, y)
print("Output:\n", output.round())
