import numpy as np 

class softMax: 

    def __init__(self, input_units, output_units): 
        # Initiallize weights and biases
        self.weights = np.random.randn(input_units, output_units) / input_units
        self.bias = np.zeros(output_units)

    def forward(self, image): 
        # stored for backprop
        self.last_image_shape = image.shape  

        # Flatten the image
        input = image.flatten()
        self.last_image = input 

        totals = np.dot(input, self.weights) + self.bias
        self.last_totals = totals 

        # Apply softmax activation
        exp = np.exp(totals)
        return exp / np.sum(exp, axis = 0)

    def backprop(self, dE_dY, alpha): 
        for i, gradient in enumerate(dE_dY): 
            if gradient == 0: 
                continue 

            z_exp = np.exp(self.last_totals)

            S = np.sum(z_exp)
            # Compute gradients with respect to output (Z)
            dY_dZ = -z_exp[i]*z_exp / (S**2)
            dY_dZ[i] = z_exp[i]*(S - z_exp) / (S**2)

            dE_dZ = gradient * dY_dZ 

            # Gradients of totals against weights/biases/input
            dZ_dw = self.last_image 
            dZ_db = 1 
            dZ_dX = self.weights 

            # Gradients of loss against weights/biases/input
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis] 
            dE_db = dZ_db * dE_dZ 
            dE_dX = dZ_dX @ dE_dZ 

            # Update weights / biases
            self.weights -= alpha*dE_dw 
            self.bias -= alpha*dE_db 

            return dE_dX.reshape(self.last_image_shape)