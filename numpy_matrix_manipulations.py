# The point of this test script is to find a way to take the linear combination of the y_hat of the
# prior layer (A) and the weights of the current layer (B) in such a way that the N y_hat elements are
# being dotted with the K groupings of N elements from the list of weights. In other words, it works
# like this:
#
# [1, 2, 3] * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# becomes
# [1 * 1 + 2 * 2 + 3 * 3, 1 * 4 + 2 * 5 + 3 * 6, 1 * 7 + 2 * 8 + 3 * 9, 1 * 10 + 2 * 11 + 3 * 12]
# [14, 32, 50, 68]
#
# This is important for constructing an ANN, because the matrix transformation being applied to B makes
# computing the forward pass and backpropagation extremely easy since the dot product can be taken with
# the output from the prior layer and the weights of the current layer and the results will be used to
# compute the input to the current node in a concise manner (each resulting element in the numpy array
# will represent a node within the current layer). For backpropagation, all you have to do is pass each
# numpy array from the nested numpy array (representing the weights) back to the prior layer and each
# numpy array will be the part of the chain rule that's d(x_curr_layer) / d(y_hat_prior_layer).

import numpy as np

# This is sample code that will be modified to determine the x (input)
# for each node in the current layer by performing a forward pass.

# Assuming A has N components and B has N * K components
N = 3  # Replace with the actual size of A
K = 4  # Replace with the actual size of B / N

# There should be a bias for every node in the current layer
bias = np.array([-1,-2,-3,-4])

# This represents the prior layer's y_hat output values
A = np.array([1, 2, 3])

# This represents the weights of the current layer
B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Reshape B into a 2D array with shape (N, K)
B_reshaped = B.reshape(K, N).T

# Use the dot function on each pair of corresponding elements
result = np.dot(A, B_reshaped)

# Add in the bias to the linear combination of the prior y_hat
# values and the current weights.
result += bias

print(A)
print(B)
print(B_reshaped)
print(result)
print("\n\n\n")

#################################################################################

# The point of this is to easily compute dL / d(y_hat_curr_layer) (if the layer is the output layer) which
# represents A, d(y_hat_curr_layer) / d(x_curr_layer) which represents B and the weights W grouped by
# sub-arrays representing the weights beloning to a specific node in the prior layer. The goal is to compute
# dL / d(y_hat_curr_layer) * d(y_hat_curr_layer) / d(x_curr_layer) * (N subarrays of length M)
# [where N = # of nodes in the prior layer, M = # of nodes in the current layer]. Multiplying the first two
# products by the N subarrays of length M is computed by doing a linear combination of the element-wise product
# of the first two arrays with the N elements in each subarray. This entire result is passed back to the prior
# layer as the current layer's loss.

# This is sample code that will be modified to compute and pass the loss, L,
# to the prior layer in the network.

# Note that the W array should be the reshaped/transposed matrix from the prior
# code sample above that was computed during the forward pass

# Example arrays A, B, and W
A = np.array([1, 2])  # Replace with the actual values of A
B = np.array([3, 4])  # Replace with the actual values of B
W = np.array([[-5,-6], [-7, -8], [-9, -10]])  # Replace with the actual values of W

# Calculate the linear combination C = A * B
C = A * B

# Multiply each element of W by the corresponding element of B
L = W * C

# Replace each sub-array in L with the sum of its elements
L = np.sum(L, axis=1)

print(A)
print(B)
print(C)
print(L)
print("\n\n\n")

#################################################################################

# NOTE Ignore this snippet of code as it's frankly useless since it doesn't take into
# account the fact that you have to multiply several arrays together.

# This code is for computing d(y_hat_curr) / d(weight_i) by distributing the value at
# each index in 'b' (the prior output value) into the sub-array at index i within matrix 'a'
# (representing the weights of the current layer)

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

b = np.array([1, 2, 3])

# Update 'a' as specified
a = a - 2 * b[:, np.newaxis]  # Adding a new axis to 'b' to make it compatible for broadcasting

print(a)
print(b[:, np.newaxis])
print(b)
print("\n\n\n")

#################################################################################

# The following code is how to distribute the general error of a node into
# the y_hat_prior array in order to calculate the weight update for a each layer.
# You're essentially foiling each element in array B into array A and constructing
# another sub-array after each number is foiled. The calculation would be as follows:
#
# C = [[1 * 0, 2 * 0],
#      [1 * 1, 2 * 1],
#      [1 * (-1), 2 * (-1)]]

# Represents the general error of the current layer for each node
# (if it's an output layer, this is d(L) / (d_y_hat_curr) * d(y_hat_curr) / d(x_curr)
#  or if it's a hidden layer, d(y_hat_curr) / d(x_curr))
A = np.array([1, 2])

# Represents the y_hat_prior values (the left layer's output) for each node
B = np.array([0, 1, -1])

# Represents the total weight error that must then be multiplied by alpha and subtracted
# from each of the weights in order to update them.
C = B[:, np.newaxis] * A

print(A)
print(B)
print(C)
print("\n\n\n")

#################################################################################

# The following code is how to compute the final error calculation for the current
# layer (the error which is to be passed on to subsequent layers to further the
# backpropagation process)

# Represents the general error of the current layer for each node
# (See the code snippet directly above)
A = np.array([1, 2])

# Represents the weights for the current layer
B = np.array([[3,4],
              [5,6],
              [7,8]])

# Represents the total error of the current layer and should be passed backwards
# as input to prior nodes to calculate their errors
C = np.dot(B, A)

print(A)
print(B)
print(C)


#################################################################################

# The following code represents how the weight loss is calculated for a hidden/output
# layer.


# This represents dL / d(y_hat_curr) * d(y_hat_curr) / d(x_curr) [i.e. general_error]
A = np.array([[1, 2, 3], [4, 5, 6]])

# This represents prior_output_val (the output values from the prior layer)
B = np.array([[-1, -2], [-3, -4]])

# This represents the matrix (same dimensions as the weight matrix) corresponding to
# the loss at each weight.
C = np.dot(B.T, A)

print(C)