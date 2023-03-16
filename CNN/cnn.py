# Source: https://www.youtube.com/watch?v=Lakz2MoHy6o

# Valid cross-correlation: len(output) = len(input) - len(kernel) + 1
# Full cross-correlation: len(output) = len(input) + len(kernel) - 1

# Dimensions:
# Input X: W x H x F
# Each kernel K: W' x H' X F each with bias B: _ x _
# Output: _ x _ x #(kernels), bias and output dimension depends on if you're doing valid or full cross-correlation

# Yi = Bi + sum(j in F):cross-correlatin(X_j, K_ij)
    # i loops in output dim (= #(kernels)), j loops in input dim (= F)

import numpy as np
from scipy import signal

class Convolution():
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape  # Ordering of dimensions for numpy
        self.depth = depth  # #(kernels)
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)  # Valid cross-correlation
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')
        return self.output

    def backward(self, output_gradient, learning_rate):
        # TODO: Update parameters and return input gradient
        pass