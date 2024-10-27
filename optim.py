import numpy as np

class SGD:
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate

class MomentumSGD:
    '''
    Implements Momentum SGD update
    '''
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        # We use a dictionary to keep track of velocities for each weight parameter
        self.velocities = {}

    def update(self, w, d_w, learning_rate):
        '''
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        '''
        # Check if velocity for `w` already exists; otherwise, initialize it
        if id(w) not in self.velocities:
            self.velocities[id(w)] = np.zeros_like(w)

        # Update velocity: v = momentum * v - learning_rate * gradient
        v = self.velocities[id(w)]
        v = self.momentum * v - learning_rate * d_w
        self.velocities[id(w)] = v

        # Update weights: w = w + v
        updated_w = w + v
        return updated_w
