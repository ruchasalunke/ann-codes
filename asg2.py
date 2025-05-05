import numpy as np
# Sigmoid and derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)
# Input and expected output for XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(0)
input_hidden_weights = np.random.uniform(size=(2, 2))
hidden_output_weights = np.random.uniform(size=(2, 1))

# Biases
hidden_bias = np.random.uniform(size=(1, 2))
output_bias = np.random.uniform(size=(1, 1))

# Training
lr = 0.5
for epoch in range(10000):
    # Forward
    hidden_input = np.dot(X, input_hidden_weights) + hidden_bias
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, hidden_output_weights) + output_bias
    final_output = sigmoid(final_input)

    # Backward
    error = y - final_output
    d_output = error * sigmoid_deriv(final_output)
    
    error_hidden = d_output.dot(hidden_output_weights.T)
    d_hidden = error_hidden * sigmoid_deriv(hidden_output)

    # Update
    hidden_output_weights += hidden_output.T.dot(d_output) * lr
    output_bias += np.sum(d_output, axis=0, keepdims=True) * lr
    input_hidden_weights += X.T.dot(d_hidden) * lr
    hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * lr

print("Final Output:\n", final_output.round())
