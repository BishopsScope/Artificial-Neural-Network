import numpy as np
from methods_forward_backward import (
    init_actv_func,
    init_loss_func,
    sigmoid,
    softmax,
    MSE,
    CCE,
)
from math import exp

class InputLayer:
    '''
    This class defines a single input layer within a neural network.
    '''

    def __init__(self, num_inputs): # input_val = np.array([])
        '''
        This initializer does nothing, since the output from an input
        layer is simply the input that was provided. If the input wasn't
        provided by the Network class, the layer will only consist of
        one node and the input to the node is zero.
        Here's what the parameter does:
        num_inputs - How many inputs the current layer can receive. This has nothing
                     to do with the batch size 
        
        input_val (old) - An array that represents the linear combination of the prior
                    layer's y_hat values and current weights + bias (or it's explicitly
                    defined by the user if the layer is an input layer)
        '''

        # Initialize the input with zeros
        self.input_val = np.zeros(num_inputs) # input_val

    def set_input(self, new_input_val):
        '''
        This method sets the value of the input(s) for the current layer and is
        utilized for the input layer ONLY since the hidden/output layers
        are set dynamically as the input is passed from layer to layer.
        '''

        self.input_val = new_input_val

    def get_input(self):
        '''
        In the context of input layers, this method simply returns the data
        that was plugged into the ANN. In the context of hidden layers and
        output layers this method is useful for making sure that the linear
        combination of weights/inputs + bias was calculated correctly (before
        the activation function is applied).
        '''

        return self.input_val

    def get_output(self):
        '''
        In the context of input layers, this method simply returns self.input_val
        but both HiddenLayers and OutputLayers override this to return self.output_val
        since a computation was actually performed.
        '''

        return self.input_val

class HiddenLayer(InputLayer):
    '''
    This class defines a single hidden layer within a neural network.
    '''

    def __init__(self, num_prior_outputs, num_inputs, weight = np.array([]), \
                 bias = np.array([]), actv_func = "Sigmoid", alpha = 0.9):
        '''
        The purpose of this initializer is to configure a hidden layer within the ANN.
        Here's what each parameter does:
        num_prior_outputs - This represents how many output nodes there are in the prior layer
                             (note that an alternative way to write this class is to allow the
                             Network class to supply the output values upon initialization, but
                             that would require the layers to be constructed during the first
                             forward pass and would require the Network class to act asymmetrically
                             since the future forward passes would have to be coded differently,
                             so it's easier to initialize num_outputs to a default value rather
                             than performing a forward pass upon initialization)
        num_inputs - This essentially defines how many nodes there are in the current layer
        weight - This is an array of weights that are associated with the y_hat values
                from the prior layer that plug into the current layer
        bias - This is an array of bias values that are used to influence input_val and
               have num_inputs number of values within the array
        actv_func - This is the activation function for the current layer. Currently supports:
                    1) Sigmoid
        alpha - This is the value used to fine-tune each node during backpropagation
        '''

        # Store the number of outputs of the prior layer
        self.num_prior_outputs = num_prior_outputs

        # Initialize an array with default values representing the output values from the
        # prior layer (this will have to be updated upon receiving a forward pass)
        self.prior_output_val = np.zeros(num_prior_outputs)

        # Store the number of nodes in the current layer
        self.num_inputs = num_inputs

        # Initialize input_val by setting each input to a default value of zero
        # (since nothing has been plugged into the current layer yet)
        #super().__init__(np.zeros(num_inputs))
        super().__init__(num_inputs)

        # If weight hasn't been initialized by the Network class, create an array
        # and initialize it to a default value
        if len(weight) == 0:

            # This is an array of default values being plugged in for every weight
            # of the current layer
            self.weight = np.ones(num_prior_outputs * num_inputs)

        else:

            # If the Network class initialized the weights, use that instead
            self.weight = weight

            # try:

            #     # If the length of the weight array isn't num_prior_outputs * num_inputs,
            #     # throw an exception
            #     if len(self.weight) != num_prior_outputs * num_inputs:

            #         raise Exception("Error! The weights provided to the layer \
            #                         don't have the correct number of values!")

            # except:

            #     raise Exception("Error! The data type you provided for weights isn't \
            #                     a numpy array!")
        
        print("Initial weight:", self.weight)
        print("Num prior outputs:", self.num_prior_outputs)
        print("Num inputs:", self.num_inputs)

        # Transform the weight array into a 2D matrix so that forward passes and
        # backpropagation can be calculated easily. The new numpy array will be
        # a 2D array with num_prior_outputs rows (arrays) by num_inputs columns
        # (number of sub-arrays).
        # self.weight = self.weight.reshape(num_inputs, num_prior_outputs).T
        self.weight = self.weight.reshape(num_prior_outputs, num_inputs)

        # If bias hasn't been initialized by the Network class, create an array
        # and initialize it to a default value
        if len(bias) == 0:

            # Note that there should be as many bias values as there are nodes
            # within the current layer
            self.bias = np.ones(num_inputs)

        else:

            # If the Network class initialized the biases, use that instead
            self.bias = bias

            # try:

            #     # If the length of the bias array isn't num_inputs, throw and exception
            #     if len(self.bias) != num_inputs:

            #         raise Exception("Error! The weights provided to the layer \
            #                         don't have the correct number of values!")
            
            # except:
                
            #     raise Exception("Error! The data type you provided for weights isn't \
            #                     a numpy array!")
        
        # # Save the activation function specifics
        # self.actv_func = actv_func

        # Determine the necessary functions to be called when performing
        # forward passes/backpropagation calculations given actv_func.
        self.forward_pass, self.d_y_hat_curr__d_x_curr = init_actv_func(actv_func)

        # Save the alpha value
        self.alpha = alpha

        # The value of output_val should never be provided by the Network class
        # explicitly, since it's the duty of the specific layer to calculate its
        # output based off of its input values and its activation function.
        # Note that the number of output values should equal the number of nodes
        # within the current layer.
        self.output_val = np.zeros(num_inputs)

        # This variable exists only for debugging purposes to see what the calculated
        # loss was, but it takes up space so it could be deleted in the future if needed.
        # It represents the array of loss values that are reported to the subsequent layer.
        self.loss = None
    
    def get_weight(self):
        '''
        This method is for viewing the weight of the current layer
        '''

        return self.weight
    
    def get_bias(self):
        '''
        This method is for viewing the bias of the current layer
        '''

        return self.bias

    def get_output(self):
        '''
        This method is for seeing what the output was after performing the forward pass.
        This is crucial for the Network class to call so that it can move the output from
        one layer to another.
        '''
        
        return self.output_val
    
    def get_loss(self):
        '''
        This method is primarily for debugging purposes to ensure that the final loss
        that will be returned to subsequent layers is calculated correctly.
        '''

        return self.loss

    def perform_forward_pass(self, prior_outputs):
        '''
        This method takes in a numpy array of the outputs from the prior layer
        (Note: the length of prior_outputs must equal self.num_prior_outputs),
        dots each value with its corresponding weight and adds the bias value
        for each particular node in the current layer. If there's an activation
        function present, it will be applied to the output. All updated arrays
        will be stored within this class, since returning an array after each
        forward pass is inefficient.
        '''

        # Save the prior outputs
        self.prior_output_val = prior_outputs

        # Make sure that the length of prior_outputs is self.num_prior_outputs
        # if len(self.prior_output_val) != self.num_prior_outputs:

        #     raise Exception("Error! The number of inputs to the current layer \
        #                     (the number of outputs from the prior layer) doesn't \
        #                     match the dimensions originally given to the current layer!")

        # Dot the prior layer's outputs with the current layer's weights
        # and add the bias value
        self.input_val = np.dot(self.prior_output_val, self.weight) + self.bias

        # Feed input_val through the forward_pass method (activation function) and assign
        # the results to output_val.
        self.output_val = self.forward_pass(self.input_val)

    def perform_backpropagation(self, total_loss):
        '''
        This method calculates the backpropagation loss for the current layer
        (this method must be overridden by OutputLayer) and multiplies it by
        the loss outputted by the layer to the right and then does the following:

        1) Calculate the loss to be used in the left layer and return it
           to the Network class to be recursively passed on through the other layers
        2) Update the weights attached to the current layer

        Note that this method assumes that total_loss is a single (vertical)
        array and isn't oriented column-wise with each value being its own sub-array.
        The number of elements in total_loss must equal the total number of nodes
        in the current layer since each node receives a loss.
        '''

        # For hidden layers, you need to update the weights as following:
        #
        # weight_i = weight_i - alpha * (total loss) * (d(y_hat_curr) / d(weight_i))
        #
        # where the term d(y_hat_curr) / d(weight_i) = y_hat_prior_i is the y output
        # from the left layer that corresponds to weight_i.
        #
        # Note that each y_hat_prior_i index corresponds to the index of the sub-array
        # within self.weight that denote which value to use when updating each sub-array
        # of weights.

        # Check to make sure the total_loss has the correct number of elements
        # if len(total_loss) != self.num_inputs:

        #     raise Exception("Error! The loss I received doesn't match the dimensions \
        #                     of my layer!")

        # Compute the general error (in the case of the hidden layer,
        # this is just total_loss * d(y_hat_curr) / d(x_curr) for the current layer).
        # General error is the overlap of error that can be used in the calculation
        # of weight/bias error as well as the error that will be passed to subsequent layers.
        general_error = total_loss * self.d_y_hat_curr__d_x_curr(self.output_val)

        # This is how many input samples were provided in the current batch
        num_input_samples = len(total_loss)

        print("Info\n###############\n")
        print("Total loss:", total_loss)
        print("d(y_hat_curr) / d(x_curr):", self.d_y_hat_curr__d_x_curr(self.output_val))
        print("Current weights:", self.weight)
        print("Current bias:", self.bias)
        print("General error:", general_error)
        print("Prior output val:", self.prior_output_val)
        print("\n###############")

        # Store the total loss calculation (this is what the Network class will need
        # for subsequent layers)
        # self.loss = np.dot(self.weight, general_error) # For a single input
        self.loss = np.dot(general_error, self.weight.T) # For multiple inputs

        print("Loss (for subsequent layers):", self.loss)
        #print("General Error x Prior Output Val:", general_error * self.prior_output_val)
        print("Prior output val (transposed):", self.prior_output_val.T)
        print("General Error x Prior Output Val:", np.dot(self.prior_output_val.T, general_error))
        #print("Mean:", np.mean(general_error * self.prior_output_val, axis=0)[:, np.newaxis])

        # Update the weight values
        # self.weight -= (self.alpha * np.mean(general_error * self.prior_output_val, axis=0)[:, np.newaxis])
        self.weight -= (self.alpha * np.dot(self.prior_output_val.T, general_error) / num_input_samples)

        # Update the bias values
        # Note that general_error is enough information to update the bias values
        # since d(x_curr) / d(bias) always produces 1
        self.bias -= (self.alpha * np.mean(general_error, axis=0))

        print("New weights:", self.weight)
        print("New bias:", self.bias)

        # # Compute d(y_hat_curr) / d(x_curr) for the current layer
        # actv_func_error = self.d_y_hat_curr__d_x_curr(self.output_val)

        # # Update all of the weights
        # self.weight -= (self.alpha * total_loss * actv_func_error *
        #                 self.prior_output_val[:, np.newaxis])

        # # Return the loss to the left layer
        # return total_loss * actv_func_error


class OutputLayer(HiddenLayer):

    def __init__(self, num_prior_outputs, num_inputs, weight = np.array([]), \
                 bias = np.array([]), actv_func = "Sigmoid", alpha = 0.9, loss_func = "MSE"):
        '''
        This initializer calls the initializer for HiddenLayer and then
        utilizes the following:
        loss_func - the loss function used at the end of the ANN. This is needed
                    to figure out what the loss is for the output layer and what
                    d(L) / d(y_hat_curr) is for the output layer.

        Note that the OutputLayer also involves the utilization of y_true, the labeled
        ground truth value at the end of the ANN that's compared to the ANNs output in
        order to determine error for the network, but this value is provided in the
        perform_backpropagation() method rather than at initialization since this value
        won't be known until all prior layers provide their outputs.
        '''

        # Call the HiddenLayer's initializer
        super().__init__(num_prior_outputs, num_inputs, weight, bias, actv_func, alpha)

        # Get the specific functions needed for the loss function
        # (to compute error values and how to find d(L) / d(y_hat_curr)).
        self.error_values, self.d_L__d_y_hat_curr = init_loss_func(loss_func)

        # This is the variable that stores the total loss obtained by the output layer
        # after comparing its output values with the true output values.
        self.final_loss = None

    def perform_backpropagation(self, y_true):
        '''
        The main difference between this overridden method and the original method
        in HiddenLayer is that total_loss will be computed within this method and then
        passed to the parent class' method for the remaining computation.
        '''

        # Compute the d(L) / d(y_hat_curr) to determine the initial loss
        # which will be passed into the current layer and subsequent layers
        # to determine the remaining loss throughout the ANN.
        total_loss = self.d_L__d_y_hat_curr(self.output_val, y_true)

        print("dL / d(y_hat_curr):", total_loss)

        # Call the parent's method to compute the necessary backpropagation calculations
        super().perform_backpropagation(total_loss)

        # Save the overall loss obtained from applying the loss function to the predicted
        # output values and the actual output values.
        self.final_loss = self.error_values(self.output_val, y_true)
    
    def get_final_loss(self):
        '''
        This method gets the final loss that was computed by applying the loss function
        to the predicted output of the output layer and the true output.
        '''

        return self.final_loss

    '''
    1) Input node:
        * An input value -- which is initialized to a specified array by the constructor
                            and its size is specified by the Network layer
        * An output value -- the array of y_hat_curr values returned after performing a
                             forward pass (no activation function needed for the input layer)
    2) Hidden node:
        * Prior layer's output value -- this is an array of y_hat_prior values (from the prior
                                        layer) that will serve to compute the input value for
                                        the current layer.
        * An input value -- which is initialized to a default array by the constructor
                            and its size is specified by the Network layer
        * Weights -- an array that is initialized with size len(y_hat_prior) * len(x_input_curr)
                     and pre-configured values or it can be specified by the Network layer
        * Bias -- an array that is initialized with size len(x_input_curr) and pre-configured
                  values or it can be specified by the Network layer
        * Activation function -- default is Sigmoid, but can be specified by the Network layer
                                 (this influences both forward pass and backpropagation)
        * Alpha -- used for backpropagation
        * An output value -- the array of y_hat_curr values returned after performing a
                             forward pass (need an activation function for the input layer)
    3) Output node:
        * Everything from the "Hidden node"
        * A loss function - default is MSE, but can be specified by the Network layer
                            (this influences both the forward pass and backpropagation)

    '''