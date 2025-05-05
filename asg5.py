import numpy as np
import matplotlib.pyplot as plt

# Perceptron Class
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if z >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update

# Dataset for AND Gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])
p = Perceptron(input_size=2)
p.train(X, y)

# Plot decision boundary
x1 = np.linspace(-1, 2, 100)
x2 = -(p.weights[1] * x1 + p.weights[0]) / p.weights[2]

plt.plot(x1, x2, label='Decision Boundary')
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c='r' if y[i]==0 else 'g')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Perceptron Decision Region")
plt.legend()
plt.grid(True)
plt.show()
