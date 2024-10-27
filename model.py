import numpy as np
from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)

class ConvNet:
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels

        # Initialize weights with Xavier initialization for conv layers
        self.conv1_weights = Parameter(np.random.randn(conv1_channels, 3, 3, input_shape[2]) * np.sqrt(2. / (3*3*input_shape[2])))
        self.conv1_bias = Parameter(np.zeros(conv1_channels))  # Shape correction for broadcasting

        conv1_out_height = input_shape[0] - 2
        conv1_out_width = input_shape[1] - 2

        self.conv2_weights = Parameter(np.random.randn(conv2_channels, 3, 3, conv1_channels) * np.sqrt(2. / (3*3*conv1_channels)))
        self.conv2_bias = Parameter(np.zeros(conv2_channels))  # Shape correction for broadcasting

        conv2_out_height = conv1_out_height - 2
        conv2_out_width = conv1_out_width - 2

        # Xavier initialization for the fully connected layer
        self.fc_weights = Parameter(np.random.randn(n_output_classes, conv2_channels * conv2_out_height * conv2_out_width) * np.sqrt(2. / (conv2_channels * conv2_out_height * conv2_out_width)))
        self.fc_bias = Parameter(np.zeros(n_output_classes))

        # Store gradients
        self.gradients = {}

    def params(self):
        return {
            'conv1_weights': self.conv1_weights,
            'conv1_bias': self.conv1_bias,
            'conv2_weights': self.conv2_weights,
            'conv2_bias': self.conv2_bias,
            'fc_weights': self.fc_weights,
            'fc_bias': self.fc_bias,
        }

    def compute_loss_and_gradients(self, X, y):
        # Clear gradients
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        # Forward pass
        out1 = self.conv_forward(X, self.conv1_weights.value, self.conv1_bias.value)
        out2 = self.conv_forward(out1, self.conv2_weights.value, self.conv2_bias.value)

        # Flatten the output for the fully connected layer
        out_flat = out2.reshape(out2.shape[0], -1)

        # Calculate scores
        scores = out_flat @ self.fc_weights.value.T + self.fc_bias.value

        # Compute softmax loss and gradient
        loss, dscores = self.softmax_loss(scores, y)

        # Backward pass
        self.backward(X, dscores, out_flat, out2, out1)

        return loss

    def conv_forward(self, X, weights, bias):
        batch_size, height, width, channels = X.shape
        n_filters, filter_height, filter_width, _ = weights.shape
        out_height = height - filter_height + 1
        out_width = width - filter_width + 1
        out = np.zeros((batch_size, out_height, out_width, n_filters))

        for b in range(batch_size):
            for f in range(n_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        out[b, i, j, f] = np.sum(X[b, i:i + filter_height, j:j + filter_width, :] * weights[f]) + bias[f]

        return out

    def softmax_loss(self, scores, y):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(len(y)), y]))

        dscores = probs
        dscores[np.arange(len(y)), y] -= 1
        dscores /= len(y)

        return loss, dscores

    def backward(self, X, dscores, out_flat, out2, out1):
        self.gradients['fc_weights'] = dscores.T @ out_flat
        self.gradients['fc_bias'] = np.sum(dscores, axis=0)

        d_out_flat = dscores @ self.fc_weights.value
        d_out2 = d_out_flat.reshape(out2.shape)

        d_conv2_weights, d_conv2_bias, d_out1 = self.conv_backward(d_out2, out1, self.conv2_weights)
        self.gradients['conv2_weights'] = d_conv2_weights
        self.gradients['conv2_bias'] = d_conv2_bias

        d_conv1_weights, d_conv1_bias, _ = self.conv_backward(d_out1, X, self.conv1_weights)
        self.gradients['conv1_weights'] = d_conv1_weights
        self.gradients['conv1_bias'] = d_conv1_bias

    def conv_backward(self, d_out, X, weights):
        batch_size, height, width, n_filters = d_out.shape
        _, filter_height, filter_width, _ = weights.shape

        d_weights = np.zeros(weights.shape)
        d_bias = np.zeros(weights.shape[0])
        d_X = np.zeros(X.shape)

        for b in range(batch_size):
            for f in range(n_filters):
                for i in range(height):
                    for j in range(width):
                        d_weights[f] += X[b, i:i + filter_height, j:j + filter_width, :] * d_out[b, i, j, f]
                        d_bias[f] += d_out[b, i, j, f]
                        d_X[b, i:i + filter_height, j:j + filter_width, :] += weights[f] * d_out[b, i, j, f]

        return d_weights, d_bias, d_X

    def predict(self, X):
        out1 = self.conv_forward(X, self.conv1_weights.value, self.conv1_bias.value)
        out2 = self.conv_forward(out1, self.conv2_weights.value, self.conv2_bias.value)
        out_flat = out2.reshape(out2.shape[0], -1)
        scores = out_flat @ self.fc_weights.value.T + self.fc_bias.value
        probabilities = self.softmax(scores)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self):
        return self.value.T

    def __getitem__(self, item):
        return self.value[item]

    def __setitem__(self, key, value):
        self.value[key] = value  # Assign new value to the underlying array

    def __repr__(self):
        return f"Parameter(shape={self.shape})"

    def copy(self):
        return Parameter(self.value.copy())


