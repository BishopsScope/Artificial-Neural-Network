# This file is for deriving the derivatives of the
# Categorical Cross Entropy Loss function and the
# Softmax function.

import numpy as np

# These are the true values
y_true = np.array([[1,0,0],
                   [0,1,0]])

# These are the inputs to the output layer
# (before being passed into the softmax
# activation function)
x_curr = np.array([[1,2,3],
                  [-4,5,16]])

# This represents x_curr being passed into the
# softmax activation layer and is the computation
# of the forward pass.
y_predict = np.exp(x_curr)
y_predict /= np.sum(y_predict, axis=1)[:, np.newaxis]

print("x_curr:", x_curr)
print("y_predict:", y_predict)
print("y_true:", y_true)
print()


# This is the error of the categorical
# cross entropy loss function with
# respect to the softmax (activation) function
d_L__d_y_hat_curr = -y_true / y_predict

# This is the derivative of the softmax function
# with respect to the output of a particular node
# (x_curr in this case, since y_hat_curr is the
# result of the activation being applied to x_curr).
# Create an array that will represent the general error
# at the current layer by utilizing the prior loss that
# was experienced.
d_y_hat_curr__d_x_curr = np.array([np.zeros(len(x_curr[0]))
                                  for _ in range(len(x_curr))])

# 'i' represents the current node that we want to calcuate
# the error for
for i in range(len(y_predict[0])):

    print(f'i: {i}')

    # 'k' represents the softmax nodes' index whos error sum
    # we need to obtain
    for k in range(len(y_predict[0])):

        print(f'k: {k}')

        print("dy_dx:", d_y_hat_curr__d_x_curr)

        # If the node whos error we're trying to calcuate (i) doesn't
        # align with the node we're at, adjust the error formula
        if i != k:

            # Modify the overall error for node 'i'
            d_y_hat_curr__d_x_curr[:, i] += (d_L__d_y_hat_curr[:, k] *
                                             (-y_predict[:, k] * y_predict[:, i]))

        # Otherwise, we use a different formula
        else:

            d_y_hat_curr__d_x_curr[:, i] += (d_L__d_y_hat_curr[:, k] *
                                             (y_predict[:, k] * (1 - y_predict[:, k])))

# general_error = d_L__d_y_hat_curr * d_y_hat_curr__d_x_curr

print("d_L__d_y_hat_curr:", d_L__d_y_hat_curr)
print("d_y_hat_curr__d_x_curr:", d_y_hat_curr__d_x_curr)
# print("General Error:", general_error)