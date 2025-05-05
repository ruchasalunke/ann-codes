import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Set the theme for seaborn for better styling of plots
sns.set_theme()

# Load the iris dataset (CSV file)
df = pd.read_csv("iris_dataset.csv")

# Prepare the input and output variables
# y ko target variable banate hain: species ke naam ko -1 (Setosa) aur 1 (baaki species) mein convert karenge
y = df.iloc[0:100].species.values
y = np.where(y == "setosa", -1, 1)  # 'setosa' ko -1 aur baaki ko 1 karenge

# Hum sirf pehle 100 rows ke first 2 features (sepal length aur sepal width) use karenge
X = df[["sepal_length", "sepal_width"]].iloc[:100].values

# Perceptron class define karte hain
class Perceptron:
    def _init_(self, eta=0.5, epochs=50):
        # eta: learning rate, jitna zyada eta, utna jaldi weight update hoga
        self.eta = eta
        self.epochs = epochs  # epochs: kitni baar training data ko iterate karna hai

    def train(self, X, y):
        # Initializing weights randomly, ek extra weight (bias) bhi add karna hai
        self.w_ = np.random.rand(1 + X.shape[1])  # 1 (bias) + features ke count
        self.errors = []  # Errors ko track karne ke liye empty list banate hain
        
        # Epochs ke liye loop
        for _ in range(self.epochs):
            errors = 0  # Har epoch mein kitni galtiyan hui hain, usko count karenge
            for xi, target in zip(X, y):  # X aur y ke elements ko ek saath iterate karenge
                # Update calculate karenge: error * learning rate
                update = self.eta * (self.predict(xi) - target)
                
                # Weights ko update karenge, except bias weight (last weight ko update karenge alag se)
                self.w_[:-1] += update * xi  # Features ke liye weight update
                self.w_[-1] += update  # Bias weight update karenge

                # Agar update != 0 hai (i.e., agar prediction galat thi), errors ko count karenge
                errors += int(update != 0)
            
            # Har epoch ke baad errors ko list mein append karenge
            self.errors.append(errors)
        
        return self  # Model ko return karenge training ke baad

    def net_input(self, X):
        # Net input calculate karte hain: X (features) aur w_ (weights) ka dot product + bias
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        # Agar net input >= 0 hai, toh 1 return karenge, else -1
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Perceptron object create kar rahe hain (learning rate = 0.1, epochs = 100)
perceptron = Perceptron(eta=0.1, epochs=100)

# Perceptron ko train kar rahe hain X aur y ke saath
perceptron.train(X, y)

# Plot Decision Regions
# Yeh plot decision boundary dikhayega, jisme classifier ka kaise output aata hai dikh raha hoga
plt.figure(figsize=(10,8))
plot_decision_regions(X, y, clf=perceptron)
plt.title("My Perceptron", fontsize=18)

plt.xlabel("Sepal Length in cm")
plt.ylabel("Sepal Width in cm")
plt.show()

# Plot Error Curve
# Yeh plot dikhayega ki har epoch ke baad perceptron ne kitni errors kiye
plt.figure(figsize=(10,10))
plt.plot(range(1, len(perceptron.errors)+1), perceptron.errors,
         marker="o", label="Error plot")
plt.xlabel("Iterations")
plt.ylabel("Missclassifications")
plt.legend()  # Plot mein legend dikhayenge
plt.show()
