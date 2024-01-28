from class_node_new import *

def main():
    '''
    NOTE This is mainly a driver method for TESTING various capabilities
    of the ANN and is NOT to be directly used as the final product since
    that's the purpose of the Network class
    '''

    # Here's the true outputs
    y_true = np.array([[1], [1]])

    input_vals = np.array([[1, 0, 1], [1, 0, 1]])

    num_output_vals = 3 # len(input_vals)

    
    input_layer = InputLayer(input_vals)

    layer_output = input_layer.get_output()

    print("Input layer's output: ", layer_output)

    print("\n\n\n")



    # Set the number of nodes in the layer
    num_nodes = 2

    # Initialize the hidden layer
    hidden_layer_1 = HiddenLayer(num_output_vals, num_nodes, \
                                 weight=np.array([0.2, -0.3, 0.4, 0.1, -0.5, 0.2]), \
                                 bias=np.array([-0.4, 0.2]))

    # Perform the forward pass
    hidden_layer_1.perform_forward_pass(layer_output)

    # View the inputs
    print("Inputs:\n", hidden_layer_1.get_input())

    # View the weight
    print("Weights:\n", hidden_layer_1.get_weight())

    # View the bias
    print("Bias:\n", hidden_layer_1.get_bias())

    # Store the output
    layer_output = hidden_layer_1.get_output()

    # Store the length of the output from the current layer.
    # Note that this value can be obtained by many different avenues,
    # but this is one way that will always work due to the nature
    # of the dimensions of the input values to the current layer.
    num_output_vals = len(hidden_layer_1.get_input()[0])

    # View the outputs
    print("Outputs:\n", hidden_layer_1.get_output())

    print("\n\n\n")



    # Set the number of nodes in the layer
    num_nodes = 1

    # Initialize the output layer
    output_layer = OutputLayer(num_output_vals, num_nodes, \
                               weight=np.array([-0.3, -0.2]), bias=np.array([0.1]))
    
    # Perform the forward pass
    output_layer.perform_forward_pass(layer_output)

    # View the inputs
    print("Inputs:\n", output_layer.get_input())

    # View the weight
    print("Weights:\n", output_layer.get_weight())

    # View the bias
    print("Bias:\n", output_layer.get_bias())

    # View the outputs
    print("Outputs:\n", output_layer.get_output())

    print("\n\n\n")


    # Perform backpropagation
    output_layer.perform_backpropagation(y_true)

    # View the updated weight
    print("Weights:\n", output_layer.get_weight())

    # View the updated bias
    print("Bias:\n", output_layer.get_bias())

    # Print the loss
    print("Loss:", output_layer.get_loss())

    # Print the final loss
    print("Final Loss:", output_layer.get_final_loss())

    print("\n\n\n")


    # Perform backpropagation on the hidden layer
    hidden_layer_1.perform_backpropagation(output_layer.get_loss())


    # input_vals = np.array([1,2,3,4,5])

    # num_output_vals = len(input_vals)


    # input_layer = InputLayer(input_vals)

    # layer_output = input_layer.generate_output()

    # print("Input layer's output: ", layer_output)

    # print("\n\n\n")


    # # How many nodes are in the hidden layer
    # num_nodes = 3

    # hidden_layer_1 = HiddenLayer(num_output_vals, num_nodes)

    # # Perform the forward pass
    # hidden_layer_1.perform_forward_pass(layer_output)

    # # View the inputs
    # print("Inputs:\n", hidden_layer_1.get_input())

    # # View the weight
    # print("Weights:\n", hidden_layer_1.get_weight())

    # # View the bias
    # print("Bias:\n", hidden_layer_1.get_bias())

    # # Store the output
    # layer_output = hidden_layer_1.get_output()

    # # Store the length of the output
    # num_output_vals = len(hidden_layer_1.get_output())

    # # View the outputs
    # print("Outputs:\n", hidden_layer_1.get_output())

    # print("\n\n\n")



    # # How many nodes are in the hidden layer
    # num_nodes = 2

    # hidden_layer_2 = HiddenLayer(num_output_vals, num_nodes)

    # # Perform the forward pass
    # hidden_layer_2.perform_forward_pass(layer_output)

    # # View the inputs
    # print("Inputs:\n", hidden_layer_2.get_input())

    # # View the weight
    # print("Weights:\n", hidden_layer_2.get_weight())

    # # View the bias
    # print("Bias:\n", hidden_layer_2.get_bias())

    # # Store the output
    # layer_output = hidden_layer_2.get_output()

    # # Store the length of the output
    # num_output_vals = len(hidden_layer_2.get_output())

    # # View the outputs
    # print("Outputs:\n", hidden_layer_2.get_output())

    # print("\n\n\n")


    # # Assume that this is the loss yielded by the output layer:
    # total_loss = np.array([1, 1])

    # # Initiate backpropagation for hidden layer 2
    # hidden_layer_2.perform_backpropagation(total_loss)

    # # View the updated weight
    # print("Weights:\n", hidden_layer_2.get_weight())

    # # View the updated bias
    # print("Bias:\n", hidden_layer_2.get_bias())

    # # Get the loss for the next layer
    # total_loss = hidden_layer_2.get_loss()

    # # View the loss that was computed for that layer
    # print("Loss:\n", total_loss)

    # print("\n\n\n")


    # # Initiate backpropagation for hidden layer 1
    # hidden_layer_1.perform_backpropagation(total_loss)

    # # View the updated weight
    # print("Weights:\n", hidden_layer_1.get_weight())

    # # View the updated bias
    # print("Bias:\n", hidden_layer_1.get_bias())

    # # Get the loss for the next layer
    # total_loss = hidden_layer_1.get_loss()

    # # View the loss that was computed for that layer
    # print("Loss:\n", total_loss)


if __name__ == '__main__':

    main()