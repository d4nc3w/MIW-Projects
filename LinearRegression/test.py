import numpy as np

y = np.array([10, 20, 30])

print("Original shape:", y.shape)
print("Reshape:", y.reshape(-1, 1).shape)