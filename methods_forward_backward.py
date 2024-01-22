import numpy as np

# The goal of this file is to be imported into class_node_new.py and to be utilized to
# define and determine the corresponding forward pass/backpropagation methods to use
# depending on what the Network class requested from class_node_new.py.

def init_actv_func(actv_func):
    '''
    This method has the duty of taking actv_func, a string describing the activation function,
    and returns a list of functions that compute the following:

        [the activation function to be called to compute forward passes,
         the derivative of current output wrt. current input (i.e. d(y_hat_curr) / d(x_curr))]
    '''

    if actv_func == "Sigmoid":

        forward_pass, d_y_hat_curr__d_x_curr = sigmoid()
    
    # Add elif statements in here for extra activation functions

    else:

        raise Exception("Error! Activation function is invalid!")

    # Return the functions that were returned from calling the activation/loss functions
    return np.array([forward_pass, d_y_hat_curr__d_x_curr])


def init_loss_func(loss_func):
    '''
    This method has the duty of taking loss_func, a string describing the loss function,
    and returns a list of functions that compute the following:

        [the loss function to be called at the start of backpropagation,
         the derivative of the loss function wrt. current output (i.e. dL / d(y_hat_curr))]
    '''

    if loss_func == None:

        error_values, d_L__d_y_hat_curr = None, None

    elif loss_func == "MSE":

        error_values, d_L__d_y_hat_curr = MSE()

    # Add elif statements in here for extra loss functions
        
    else:

        raise Exception("Error! Loss function is invalid!")
    
    # Return the functions that were returned from calling the activation/loss functions
    return np.array([error_values, d_L__d_y_hat_curr])

################################################################################################

# Below are the activation functions

def sigmoid():
    '''
    This method returns a numpy array containing two functions which can
    be called as many times as needed when a layer of the ANN is performing
    a forward pass/backpropagation. The returned numpy array is for as follows:

        [takes the current input from the layer (x_curr) and produces the output (y_hat_curr),
         computes the derivative of current output wrt. current input]
    '''

    def forward_pass(x_curr):

        return 1 / (1 + np.exp(-x_curr))

    def d_y_hat_curr__d_x_curr(y_hat_curr):

        return y_hat_curr * (1 - y_hat_curr)

    # Returns the functions necessary for calculating future forward passes
    # and computing the derivative of the Sigmoid activation function.
    return np.array([forward_pass, d_y_hat_curr__d_x_curr])


def softmax():
    '''
    This method applies the Softmax activation function to the input_val numpy array
    and returns its output. It also returns d(y_hat_curr) / d(x_curr) since that's the
    part of the chain rule for backpropagation pertaining to the activation function's
    calcuation.
    '''

    pass

################################################################################################

# Below are the loss functions

# NOTE: these methods shouldn't be coded until HiddenLayer is finished and you're working
# on OutputLayer!

def MSE():
    '''
    This method yields two functions that calculate the following:
    
        [the error value for each node in the output layer,
         the derivative of the loss wrt. current output (i.e. d(L) / d(y_hat_curr))]
    '''

    def error_values(y_hat_curr, y_true):
        '''
        The error for MSE is computed as the following:

            Loss = (1 / (2N)) * summation[i=1 -> N]((y_hat_curr_i - y_true_i)^2)

            d(L) / d(y_hat_curr) = (1 / N) * (y_hat_curr_i - y_true_i)
                                 = (1 / N) * (y_hat_curr - y_true)  [in matrix notation]
        '''

        # Make sure the dimensions of y_hat_curr and y_true are the same
        if len(y_hat_curr) != len(y_true):

            raise Exception("Error! The dimensions of the output layer's output \
                             and the dimensions of the true output aren't the same!")

        # Store the number of elements
        N = len(y_hat_curr)

        # Return the error (i.e. a single value to be reported which outlines
        # the ANNs overall error)
        return (1 / (2 * N)) * np.sum(np.square(y_hat_curr - y_true))

    def d_L__d_y_hat_curr(y_hat_curr, y_true):

        # Make sure the dimensions of y_hat_curr and y_true are the same
        if len(y_hat_curr) != len(y_true):

            raise Exception("Error! The dimensions of the output layer's output \
                             and the dimensions of the true output aren't the same!")

        # Store the number of elements
        N = len(y_hat_curr)

        # Return the derivative of the loss wrt. output layer
        return (y_hat_curr - y_true) / N
    
    # Return the corresponding functions in list format
    return np.array([error_values, d_L__d_y_hat_curr])


def CCE():
    '''
    This method calculates the loss function, Categorical Cross Entropy, using the output
    values from the output layer and the y_true values (i.e. the correct labels used
    to train the network).
    '''

    pass