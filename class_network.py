from class_node_new import *

class Network:
    '''
    The purpose of the Network class is to receive ANN
    specifications from the driver code and then to construct
    and objects from the corresponding Layer classes to match the
    specific network requirements. It's also responsible for
    receiving input data from the driver code, reporting the
    experienced losses to the driver code (if the driver code wants it),
    training the ANN and testing the ANN. It's also responsible for
    calling all of the methods within the Layer classes to perform
    forward and backward propagation and well as to retain and pass on
    the layer's output or overall loss from one layer to the next.
    '''
    
    def __init__(self, nodes_per_layer, layer_weights, \
                 layer_biases, actv_funcs, alphas, loss_func):
        '''
        The point of this constructor is to initialize the various
        layers that the driver code wants present in the ANN and to
        configure their corresponding parameters. Here's what each
        component does:
        nodes_per_layer - A numpy array where each element corresponds to
                          the number of nodes in that layer, with the value
                          at index 0 corresponding to the number of nodes in
                          the input layer and the value at the last index
                          corresponding to the number of nodes in the output
                          layer. Every layer must have a specified number of
                          nodes.
        layer_weights - A numpy array of numpy arrays corresponding to the
                        weights at each layer. There should be num_layers - 1
                        numpy arrays corresponding to weights, since the input
                        layer doesn't have any weights.
        layer_biases - A numpy array of numpy arrays corresponding to the
                       biases in each layer (excluding the input layer, since
                       the input layer doesn't have biases).
        actv_funcs - A list of activation functions for every layer that isn't
                     an input layer (since input layers don't have activation
                     functions) from left to right across the ANN.
        alphas - A numpy array with each element corresponding to the alpha value
                 for that layer, not including the input layer (since backpropagation
                 doesn't occur in the input layer).
        loss_func - A string corresponding to the loss function being applied at the
                    output layer
        '''

        # Note that the InputLayer isn't used in this class at all, because it's
        # not needed so long as you're provided with the training/testing data
        # since you can simply plug that into the network without needing a class
        # to define it since there's no activation function, loss function, alpha
        # value or anything else that's applied to the input layer in any way.

        # NOTE: When the Network class is given training/testing data, that data
        # should be passable in its entirety or in chunks (mini-batches) to each
        # individual layer within the network for computation. This serves several
        # purposes:
        #
        # 1) The dot products being computed only once per forward pass with the
        #    entire dataset utilize the ability of Numpy to work with large data.
        # 2) You can compute the average loss at the end of the forward pass for
        #    the entire training set and backpropagate its loss in one single
        #    iteration rather than computing it for every example, where some
        #    examples may present noise due to them being outliers and hence
        #    the gradient updates would be influenced much more by them if they're
        #    being passed through the network as individual inputs rather than
        #    collective inputs.

        # Store how many layers there are (input, hidden and output)
        self.num_layers = len(nodes_per_layer) # num_layers

        # Step 1) Save the number of inputs required for the input layer

        self.num_input_nodes = nodes_per_layer[0]

        # Step 2) Construct the hidden layers

        # print("Details:")
        # print(f"{nodes_per_layer[0]}\n{nodes_per_layer[1]}\n{layer_weights[0]}\n \
        #       {layer_biases[0]}\n{actv_funcs[0]}\n{alphas[0]}")

        self.hidden_layers = np.array([HiddenLayer(nodes_per_layer[i], nodes_per_layer[i+1],
                                                   layer_weights[i], layer_biases[i],
                                                   actv_funcs[i], alphas[i])
                                       for i in range(self.num_layers - 2)])

        # Step 3) Construct the output layer

        self.output_layer = OutputLayer(nodes_per_layer[-2], nodes_per_layer[-1],
                                        layer_weights[-1], layer_biases[-1],
                                        actv_funcs[-1], alphas[-1], loss_func)

    def perform_training(self, input_data, output_data, num_epochs=1):
        '''
        This method performs training for the neural network by passing
        input data through the ANN, calculating error within the output data,
        performing backpropagation to update weights/biases and then repeating
        the process for the number of epochs specified.
        input_data - The input values to be passed to the input layer of the ANN
                     in the form of a numpy array of numpy arrays with each numpy
                     sub-array representing a single array of inputs for the input
                     layer of the ANN.
        output_data - The labels for the data (y_true values) for each input sample
                      in the form of a numpy array of numpy arrays with each sub-array
                      representing the array of true output values.
        num_epochs - How many times the same input will be passed through the network
                     for training purposes to update the weights/biases.
        '''

        # Loop through the number of epochs for training
        for i in range(num_epochs):

            # input_data represents the output from the input layer, so pass the
            # input_data into the forward pass of the first hidden layer.
            self.hidden_layers[0].perform_forward_pass(input_data)

            # For every hidden layer after the first hidden layer,
            # perform forward propagation
            for layer_index in range(1, self.num_layers - 2):

                # Perform the forward pass on the current hidden layer based off
                # of the output from the previous hidden layer.
                self.hidden_layers[layer_index].perform_forward_pass(
                    self.hidden_layers[layer_index - 1].get_output()
                )
            
            # Compute the forward pass for the output layer based off of the output
            # from the last hidden layer.
            self.output_layer.perform_forward_pass(
                self.hidden_layers[-1].get_output()
            )

            # Print the output of the network
            print(f"Output for input #{i}")
            print(f"{input_data}\n\n{self.output_layer.get_output()}")

            # Perform backpropagation for the output layer on the output data
            self.output_layer.perform_backpropagation(output_data)

            print("\nPerforming backpropagation through the network...\n")

            # Perform backpropagation on the last hidden layer with the loss from
            # the output layer
            self.hidden_layers[-1].perform_backpropagation(self.output_layer.get_loss())

            # Perform backpropagation on the remaining hidden layers
            # from the second to last hidden layer to the first hidden layer
            for layer_index in range(self.num_layers - 4, -1, -1):

                # Perform backpropagation by taking in the error from the next
                # hidden layer and updating the weights/biases for the current
                # hidden layer
                self.hidden_layers[layer_index].perform_backpropagation(
                    self.hidden_layers[layer_index + 1].get_loss()
                )

            print("\n\nForward and backward pass complete!\n\n")


    def perform_testing(self, input_data):
        '''
        This method passes input data through the ANN and acts very similarly to
        perform_training(), except that it doesn't update the network and simply
        computes the forward pass and returns the output for every input within
        the input data.
        input_data - The input values to be passed to the input layer of the ANN
                     in the form of a numpy array of numpy arrays with each numpy
                     sub-array representing a single array of inputs for the input
                     layer of the ANN. 
        '''

        pass