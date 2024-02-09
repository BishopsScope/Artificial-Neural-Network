from class_network import *

def main():

    # Build the network
    network = Network(nodes_per_layer=np.array([1, 2, 1]),
                      layer_weights=np.array([np.array([]),
                                              np.array([]),
                                              np.array([]),
                                              np.array([]),
                                              np.array([])]),
                      layer_biases=np.array([[], [], [], [], []]),
                      actv_funcs=['Sigmoid', 'Sigmoid', 'Sigmoid', 'Sigmoid', 'Sigmoid'],
                      alphas=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                      loss_func='MSE')
    
    # Train the network
    network.perform_training(input_data=np.array([[1], [0]]),
                             output_data=np.array([[0], [1]]),
                             num_epochs=1)


if __name__ == '__main__':

    main()