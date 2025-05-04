import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Create 1000 evenly spaced values between -10 and 10
x = np.linspace(-10,10,1000) 
def linear_function(x):
    return x
def sigmoid_function(x):
    return 1 / (1+np.exp(-x))
def tanh_function(x):
    return np.tanh(x)
def relu_function(x):
    return np.maximum(0,x)
linear = linear_function(x)
sigmoid = sigmoid_function(x)
tanh = tanh_function(x)
relu = relu_function(x)
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.plot(x,linear)
plt.title('Linear')
plt.subplot(2,2,2)
plt.plot(x,sigmoid)
plt.title('Sigmoid')
plt.subplot(2,2,3)
plt.plot(x,tanh)
plt.title('Tanh')
plt.subplot(2,2,4)
plt.plot(x,relu)
plt.title('ReLU')
plt.tight_layout()
plt.show()
