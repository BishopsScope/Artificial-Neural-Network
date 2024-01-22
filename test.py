import numpy as np

# Assuming A has N components and B has N * K components
N = 3  # Replace with the actual size of A
K = 4  # Replace with the actual size of B / N

# Example arrays A and B
A = np.array([1, 2, 3])
B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Replace with the actual values of B

# Reshape B into a 2D array with shape (N, K)
B_reshaped = B.reshape(K, N).T

# Use the dot function on each pair of corresponding elements
result = np.dot(A, B_reshaped)

print(A)
print(B_reshaped)
print(result)