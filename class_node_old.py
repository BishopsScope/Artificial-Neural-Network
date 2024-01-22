from math import exp

class Node:
    '''
    This is the class that defines each node within
    an artificial neural network.
    '''
    
    def __init__(self, alpha=0.5, input_val=0, bias=None, weights=[], actv_func="Sigmoid"):

        # Store the type of activation function within the node
        self.actv_func = actv_func

        # This is our alpha (step size) for the current node
        self.alpha = alpha

        # This is the input value to the node
        # and should be calculated in the Network
        # class before initializing this node.
        # This needs to be stored within the node
        # so that the node can calculate the
        # backpropagation formula.
        self.input_val = input_val

        # The current output is nothing
        self.output = None

        # This is the bias of the current node.
        self.bias = bias

        # Compute the output and save the value in self.output
        # self.compute_output()

        # This is the weights stemming out from the output
        # of the node that connects to the other nodes in
        # the layer in front of the current node. The number
        # of weights must be equal to the number of nodes
        # in front of the current node in order for the
        # network to be fully connected.
        self.weights = weights

        # The different types of nodes that can exist depending
        # on what is provided to the constructor is the following:
        # 1) Input node:
        #       Has N weights and no bias
        # 2) Hidden node:
        #       Has N weights and one bias
        # 3) Output node:
        #       Has no weights and one bias
        # If neither a weight or a bias is provided to the constructor,
        # an exception will be raised since the type of node provided
        # is invalid due to it not meeting the above criteria.

        if self.weights == [] and self.bias == None:

            raise Exception("You created an invalid node!")
        
        elif self.weights == []:

            self.node_name = "Output"

        elif self.bias == None:

            self.node_name = "Input"

        else:

            self.node_name = "Hidden"

        # The error at the current node is currently 1,
        # since nothing has happened yet and this value
        # will be multiplied by subsequent errors at
        # nodes in front of the current node via the
        # chain rule to get the total loss for weights
        # behind the current node. Since subsequent errors
        # need to be multiplied by the current error,
        # 1 is the logical choice since 1 times anything
        # is just that thing.
        self.error = 1

    def set_input(self, input_val):
        '''
        Set the initial input for a node.
        '''
        self.input_val = input_val

    def compute_output(self):
        '''
        This function computes the output
        of the current node and saves it
        in the output variable.
        '''
        # Here's where you need to know which activation
        # function is being used so you know what your
        # output is.
        if self.actv_func == "Sigmoid":

            # Return the result when input_val
            # is plugged into the activation function.
            return 1 / (1 + exp(-self.input_val))
        
        else:

            raise Exception("The activation function was incorrectly specified!")
        
    def get_error(self, next_errors, training_output):
        '''
        This function calculates the error for
        the current node given the error of the
        node immediately ahead of the current
        node. Once the error is calculated, it
        will be used within the backpropagation
        process to train every weight/bias within
        the artificial neural network.
        next_error is the sum of the error value(s)
        of the node(s) in front of the current node.
        Once this error value is computed by the
        Network class and the sum of error(s) from the
        node(s) in front of the current node are
        calculated, then all of the weights immediately
        before the current node will automatically be
        updated with that new value.
        '''

        # There are three types of error values: 1) the node's
        # error xi_error for the i-th node, 2) all weight-based
        # errors for dL/dw_i for the i-th weight for every weight
        # coming out of the current node and 3) the bias node's
        # error for dL/dbi for the i-th bias value.

        
        # The chain rule that describes the losses
        # for the nodes in front of the current node
        # is presented with next_error (which is
        # copulated by the Network class and passed
        # to every node in the network).

        # Each node is responsible for updating the
        # weights that plug into itself.

        # There's two types of losses that need to be
        # computed and returned from this function:
        # 1) next_error * (dy/dx) * (dy/dx) where
        #    dy/dx is the derivative of the current
        #    node's output with respect to the input
        #    of the current node.
        #    dx/dy is the derivative of the current
        #    node's input with respect to the output
        #    of a prior node.
        #    This is the loss that needs to be passed
        #    back to the previous layer of nodes and
        #    is NOT used to update the current node's
        #    weights.
        # 2) next_error * (dy/dx) * (dx/dw) where
        #    dy/dx is the same as in 1)
        #    dx/dw is the derivative of the current
        #    node's input with respect to the weight
        #    that's plugged into the current node
        #
        # Note that both 1) and 2) are errors which
        #      need to be returned for every node
        #      which connects to the current node,
        #      so the returned list should be N x 2
        #      in dimension where N is the number of
        #      nodes that connect to the current node
        #      and 2 is the number of errors returned
        #      for every prior connected node.

        # Compute the output of the current node
        self.output = self.compute_output()

        if self.node_name == "Output":
            
            # Calculate the current node's error
            xi_error = (self.output - training_output) * self.output * (1 - self.output)

            # There's no error for the weights, since no weights exist

            # Update the bias value
            self.bias -= (self.alpha * xi_error)

            # Return the current node's error
            return xi_error

        elif self.node_name == "Hidden":
            
            # The error for the current node is initially zero,
            # but we'll end up adding all error paths to the
            # current node by iterating through the next_errors
            # list.
            xi_error = 0

            # Iterate through the next_errors list
            for node_index in range(len(next_errors)):

                # Add one of the error paths the the overall error
                # of the current node.
                xi_error += next_errors[node_index] * self.weights[node_index] * self.output * (1 - self.output)

                # Calculate the error for the weight that connects us
                # to the path we just added to xi_error (in front of
                # our current node) so we can update the weight in
                # front of us.
                dL_dwi = next_errors[node_index] * self.output

                # Update the weight that we calculated the error for.
                self.weights[node_index] -= (self.alpha * dL_dwi)

            # Update our node's bias value
            self.bias -= (self.alpha * xi_error)

            # Return our current node's error
            return xi_error

        elif self.node_name == "Input":

            # The error for the current node is initially zero,
            # but we'll end up adding all error paths to the
            # current node by iterating through the next_errors
            # list.
            xi_error = 0

            # Iterate through the next_errors list
            for node_index in range(len(next_errors)):

                # Add one of the error paths the the overall error
                # of the current node.
                xi_error += next_errors[node_index] * self.weights[node_index] * self.output * (1 - self.output)

                # Calculate the error for the weight that connects us
                # to the path we just added to xi_error (in front of
                # our current node) so we can update the weight in
                # front of us.
                dL_dwi = next_errors[node_index] * self.output

                # Update the weight that we calculated the error for.
                self.weights[node_index] -= (self.alpha * dL_dwi)

            # There's no bias value to update, since we're an input node.

            # Return our current node's error
            return xi_error