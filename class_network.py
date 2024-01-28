import class_node_new

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
    
    def __init__(self, num_layers, nodes_per_layer, layer_weights, \
                 layer_biases, actv_funcs, alphas, loss_func):
        '''
        The point of this constructor is to initialize the various
        layers that the driver code wants present in the ANN and to
        configure their corresponding parameters. Here's what each
        component does:
        num_layers - Describes how many input, hidden and output layers
                     there are.
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
                       biases in each layer.
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

        pass