import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Implement It
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO Implement IT
    raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO Implement it
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO Implement it
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO Implement it
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO Implement it

        raise Exception("Not implemented!")
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding):
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size, in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding

        self.X = X

        # Initialize output tensor
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # Pad the input if padding is specified
        X_padded = np.pad(X, ((0, 0), (self.padding, self.padding),
                              (self.padding, self.padding), (0, 0)), mode='constant')

        # Convolutional operation
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :]
                for k in range(self.out_channels):
                    out[:, y, x, k] = np.sum(X_slice * self.W.value[:, :, :, k], axis=(1, 2, 3))

        # Add bias
        out += self.B.value

        return out

    def backward(self, d_out):
        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.W.value)
        dB = np.zeros_like(self.B.value)

        # Pad X and dX for padding during backward pass
        X_padded = np.pad(self.X, ((0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding), (0, 0)), mode='constant')
        dX_padded = np.pad(dX, ((0, 0), (self.padding, self.padding),
                                (self.padding, self.padding), (0, 0)), mode='constant')

        # Compute gradients
        batch_size, height, width, in_channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # Convolutional operation for backward pass
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :]
                for k in range(out_channels):
                    # Compute gradient with respect to weights
                    dW[:, :, :, k] += np.sum(X_slice * d_out[:, y, x, k][:, None, None, None], axis=0)
                    # Compute gradient with respect to biases
                    dB[k] += np.sum(d_out[:, y, x, k], axis=0)
                    # Compute gradient with respect to input
                    dX_padded[:, y:y + self.filter_size, x:x + self.filter_size, :] += (
                            self.W.value[:, :, :, k] * d_out[:, y, x, k][:, None, None, None]
                    )

            # Remove padding from dX to match the input shape
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded

            # Store gradients
        self.W.grad = dW
        self.B.grad = dB

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        '''
        Performs the forward pass of the max pooling layer

        Arguments:
        X, np.array - input data, shape (batch_size, height, width, channels)

        Returns:
        np.array - output data after pooling
        '''
        batch_size, height, width, channels = X.shape
        # Calculate output dimensions
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        # Initialize output tensor
        output = np.zeros((batch_size, output_height, output_width, channels))

        # Perform max pooling operation
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                output[:, i, j, :] = np.max(X[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))

        return output
        raise Exception("Not implemented!")

    def backward(self, d_out, X):
        '''
        Performs the backward pass of the max pooling layer

        Arguments:
        d_out, np.array - gradient of the loss with respect to the output of the pooling layer
        X, np.array - input data from the forward pass

        Returns:
        np.array - gradient of the loss with respect to the input of the pooling layer
        '''
        batch_size, output_height, output_width, channels = d_out.shape
        d_input = np.zeros((batch_size, output_height * self.stride + self.pool_size - self.stride,
                            output_width * self.stride + self.pool_size - self.stride, channels))

        # Backpropagation through max pooling
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                for k in range(channels):
                    # Find the maximum value's index in the region
                    mask = (X[:, h_start:h_end, w_start:w_end, k] ==
                            np.max(X[:, h_start:h_end, w_start:w_end, k], axis=(1, 2), keepdims=True))

                    # Distribute the gradient to the corresponding position in d_input
                    d_input[:, h_start:h_end, w_start:w_end, k] += d_out[:, i, j, k][:, np.newaxis, np.newaxis] * mask

        # Crop d_input to match the original input shape if padding was not applied
        return d_input[:, :X.shape[1], :X.shape[2], :]
        raise Exception("Not implemented!")

    def params(self):
        '''
        Returns an empty dictionary as max pooling has no trainable parameters
        '''
        return {}


class Flattener:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        # Flatten the input tensor
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out, x=None):
        # Reshape the gradient back to the original input shape
        return d_out.reshape(self.input_shape)