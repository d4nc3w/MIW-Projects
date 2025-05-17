import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_prepare_data(filepath, label_column_name=None):
    df = pd.read_csv(filepath)

    if label_column_name is None:
        label_column = df.columns[-1]
    else:
        label_column = label_column_name
    X_raw = df.drop(columns=[label_column])
    y_raw = df[label_column]

    def label_encode_column(column):
        unique_values = sorted(column.unique())
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        return column.map(value_to_int).to_numpy()

    X_encoded = []
    for col in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            X_encoded.append(X_raw[col].to_numpy())
        else:
            X_encoded.append(label_encode_column(X_raw[col]))
    X = np.stack(X_encoded, axis=1)
    y = label_encode_column(y_raw)

    def encode_with_one_hot_encoding(values):
        num_classes = len(np.unique(values))
        return np.eye(num_classes)[values]
    Y = encode_with_one_hot_encoding(y)
    return X.T, Y.T

X, Y = load_and_prepare_data("gender/trainSet.csv")

print(f"Shape of input: {X.shape}")
print(f"Example of input: {X[:, 0]}")
print(f"Shape of target: {Y.shape}")
print(f"Example of target: {Y[:, 0]}")

def extend_input_with_bias(network_input):
    bias_extension = np.ones(network_input.shape[1]).reshape(1, -1)
    network_input = np.vstack([bias_extension, network_input])
    return network_input

def describe_data(data):
    return f'number of features is {data.shape[0]} and number of datapoint is {data.shape[1]}'

print(f'For input {describe_data(X)}.')
print(f'For output {describe_data(Y)}.')

def create_network(input_size, output_size, hidden_sizes):
    network = []
    layer_sizes = hidden_sizes
    layer_sizes.append(output_size)
    for neuron_count in layer_sizes:
        layer = np.random.rand(input_size+1, neuron_count)*2-1
        input_size = neuron_count
        network.append(layer)
    return network

def describe_layer(layer):
    return f'there is {layer.shape[1]} neurons with {layer.shape[0]} inputs each'

network = create_network(X.shape[0], Y.shape[0], [7, 6])
for idx, layer in enumerate(network):
    print(f'In layer {idx} {describe_layer(layer)}')

def unipolar_activation(u):
    return 1/(1 + np.exp(-u))

def unipolar_derivative(u):
    a = unipolar_activation(u)
    return a * (1.0 - a)

demo_x = np.linspace(-5, 5, 100)
demo_y = unipolar_activation(demo_x)
demo_derivative = unipolar_derivative(demo_x)
plt.plot(demo_x, demo_y, label='Unipolar')
plt.plot(demo_x, demo_derivative, label='Derivative')
plt.legend()
plt.show()


def feed_forward(network_input, network):
    layer_input = network_input
    responses = []

    for weights in network:
        layer_input = extend_input_with_bias(layer_input)
        response = unipolar_activation(weights.T @ layer_input)
        layer_input = response
        responses.append(response)
    return responses

def predict(network_input, network):
    return feed_forward(network_input, network)[-1]

def calculate_mse(predicted, expected):
    return np.sum((predicted - expected) ** 2) / len(predicted)

Y_predicted = predict(X, network)
print(f"mse = {calculate_mse(Y_predicted, Y)}")

responses = feed_forward(X, network)
for idx, response in enumerate(responses):
    print(f'For response of {idx} layer {describe_data(response)}')

print(np.argmax(Y_predicted, axis=0))

def backpropagate(network, responses, expected_output_layer_response):
    gradients = []
    error = responses[-1] - expected_output_layer_response
    for weights, response in zip(reversed(network), reversed(responses)):
        gradient = error + unipolar_derivative(response)
        gradients.append(gradient)
        error = weights @ gradient
        error = error[1:,:]
    return list(reversed(gradients))

gradients = backpropagate(network, responses, Y)
for idx, gradient in enumerate(gradients):
    print(f'For gradient of {idx} layer {describe_data(gradient)}')

def calculate_weight_changes(network, network_input, network_responses, gradients, learning_factor):
    layer_inputs = [network_input] + network_responses[:-1]
    weights_changes = []
    for weights, layer_input, gradient in zip(network, layer_inputs, gradients):
        layer_input = extend_input_with_bias(layer_input)
        change = layer_input.dot(gradient.T) * learning_factor
        weights_changes.append(change)
    return weights_changes

changes = calculate_weight_changes(network, X, responses, gradients, 0.01)
for idx, change in enumerate(changes):
    print(f'For change of {idx} layer {describe_layer(change)}')

def adjust_weights(network, changes):
    new_network = []
    for weights, change in zip(network, changes):
        new_weights = weights - change
        new_network.append(new_weights)
    return new_network

network = adjust_weights(network, changes)
for idx, layer in enumerate(network):
    print(f'In layer {idx} {describe_layer(layer)}')
Y_predicted = predict(X, network)
print(f'MSE = {calculate_mse(Y_predicted, Y)}')

def train_network(network, network_input, expected_output, learning_factor, epochs):
    mse_history = []
    for _ in range(epochs):
        responses = feed_forward(network_input, network)
        mse_history.append(calculate_mse(responses[-1], expected_output))
        gradients = backpropagate(network, responses, Y)
        changes = calculate_weight_changes(network, network_input, responses, gradients, learning_factor)
        network = adjust_weights(network, changes)
    mse_history.append(calculate_mse(responses[-1], expected_output))
    return network, np.asarray(mse_history)

network = create_network(X.shape[0], Y.shape[0], [7, 6])
network, mse_history = train_network(network, X, Y, 0.3, 10)
plt.plot(mse_history)
plt.show()

network = create_network(X.shape[0], Y.shape[0], [7, 6])
network, mse_history = train_network(network, X, Y, 0.001, 20)
plt.plot(mse_history)
plt.show()