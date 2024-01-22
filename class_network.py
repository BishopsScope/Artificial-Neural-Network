import class_node_new

class Network:
    '''
    The purpose of the Network class is to receive ANN
    specifications from the driver code and then to construct
    and call the imported InputLayer, HiddenLayer and OutputLayer
    classes to match the specific network requirements.
    It's also responsible for receiving input data from the driver
    code, reporting the experienced losses to the driver code
    (if the driver code wants it), training the ANN and testing
    the ANN. It's also responsible for calling all of the methods
    within the InputLayer, HiddenLayer and OutputLayer classes
    to perform forward and backward propagation and well as to retain
    and pass on the layer's output or overall loss from one layer
    to the next.
    '''
    
    def __init__(self):
        '''
        The point of this constructor is to initialize the various
        layers that the driver code wants present in the ANN and to
        configure their corresponding parameters
        '''

        pass