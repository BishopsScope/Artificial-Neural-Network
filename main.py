import numpy as np
from class_node_new import *

def main():
    '''
    This is the driver code for creating, running,
    training, testing and retrieving the final output
    of an artificial neural network given training
    and testing data.
    '''

    # Initially you're going to want to load the data that will
    # be split into training data and testing data. Split the
    # data into training and testing data at this stage.

    # Here you would call the Network class and pass it parameters
    # that indicate the details that are used to form the network.
    # This includes the activation function that will be used for
    # each layer of nodes (each layer may have a different
    # activation function) as well as a loss function (i.e. MSE).

    # Once the Neural Network is initialized by the Network class,
    # you would call the training method within the Network class
    # with parameters indicating how many epochs will be used, the
    # data that will be used during training and any other relevant
    # parameters.

    # Once the training is completed, call the testing method
    # within the Network class and pass the testing data as a
    # parameter along with other parameters and save the
    # predicted outputs from the network and compare its output
    # to your labeled data.

    # Call get_weights() in the Network class that will function
    # by retrieving all of the weights and biases within the network
    # and return them in a concise list-based fashion so the trained
    # network can be saved.

    # Dont forget to add within the Network class the functionality
    # to choose your own weights so that previously trained neural
    # networks can easily be restored for further testing or for
    # deployment purposes.

    obj = InputLayer(np.array([1, 2, 3]))
    print(obj.generate_output())

    pass


if __name__ == '__main__':

    main()