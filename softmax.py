import numpy as np

class softMax:

    def __init__(self, input_units, output_units):
        self.weights = np.random.randn(input_units, output_units) / input_units
        self.bias = np.zeros(output_units)

    def forward(self, image):

        self.last_image_shape = image.shape
        x = image.flatten()
        self.last_input = x

        totals = np.dot(x, self.weights) + self.bias
        self.last_totals = totals

        exp = np.exp(totals - np.max(totals))     # tránh overflow
        self.out = exp / np.sum(exp)

        return self.out


    def backprop(self, dE_dY, lr):

        # Softmax Jacobian: dy/dz = y * (I - y)
        y = self.out

        # dE/dZ = Jacobian(softmax)^T * dE/dY
        # Vectorized: dE/dZ = y*(dE/dY) - y * sum(y * dE/dY)
        dE_dZ = y * dE_dY - y * np.dot(y, dE_dY)   

        # Gradients
        dE_dw = np.outer(self.last_input, dE_dZ)   # (in, out)
        dE_db = dE_dZ
        dE_dX = self.weights @ dE_dZ

        # Update weights
        self.weights -= lr * dE_dw
        self.bias -= lr * dE_db

        return dE_dX.reshape(self.last_image_shape)
