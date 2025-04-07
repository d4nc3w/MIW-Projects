import numpy as np
import matplotlib.pyplot as plt

def generate_example_data(count, noise_level):
    x = np.arange(0, count, 1, dtype='float64')
    y = 2*x + 3
    noise = (np.random.rand(count)*2-1)*noise_level
    y += noise
    return x, y

def generate_polynomial_data(count, noise_level):
    x = np.arange(0-count//2, count-count//2, 1, dtype='float64')
    y = x**2 + 6
    noise = (np.random.rand(count)*2-1)*noise_level
    y += noise
    return x, y

def calculate_normal_equation(X, y):
    tmp = X.T.dot(X)
    tmp = np.linalg.inv(tmp)
    tmp = tmp.dot(X.T)
    theta_hat = tmp.dot(y)
    return theta_hat

def calculate_hypothesis(X, theta):
    y = theta.T.dot(X.T)
    return y

#-----------------------------------------------------------------
NUMBER_OF_SAMPLES = 100
NUMBER_OF_FEATURES = 1

x, y = generate_example_data(NUMBER_OF_SAMPLES, 30)
plt.scatter(x, y, c='green')
plt.show()

print(f'Shape of x i s {x.shape}')
print(f'Shape of y i s {y.shape}')

X = x.reshape(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES)
X = np.c_[np.ones((X.shape[0],1)),X]
y = y.reshape(NUMBER_OF_SAMPLES, 1)

print(f'Shape of X i s {X.shape}')
print(f'Shape of y i s {y.shape}')

theta_hat = calculate_normal_equation(X, y)
y_hat = calculate_hypothesis(X, theta_hat).flatten()

plt.scatter(x, y, c='green')
plt.plot(x, y_hat, c='red')
plt.show()

x_squared = x**2
x_squared = x_squared.reshape(NUMBER_OF_SAMPLES, 1)

x, y = generate_polynomial_data(NUMBER_OF_SAMPLES, 0)
plt.scatter(x, y, c='green')
plt.show()

X = x.reshape(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES)
X = np.c_[np.ones((X.shape[0],1)),X, x_squared]
y = y.reshape(NUMBER_OF_SAMPLES, 1)

print(f'Shape of X i s {X.shape}')
print(f'Shape of y i s {y.shape}')
print(f'Example of input {X[3]}')

theta_hat = calculate_normal_equation(X, y)
y_hat = calculate_hypothesis(X, theta_hat).flatten()

plt.scatter(x, y, c='green')
plt.plot(x, y_hat, c='red')
plt.show()