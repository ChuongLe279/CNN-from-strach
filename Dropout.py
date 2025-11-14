import numpy as np

class Layer_Dropout:
    # Initialize the dropout layer
    def __init__(self, rate):
        # 'rate' is the dropout rate (probability to drop a unit)
        # Convert it to keep-probability for sampling
        # e.g., rate=0.1  -> keep_prob = 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
        # Save input values (sometimes useful for debugging or consistency)
        self.inputs = inputs
        # Sample a Bernoulli (0/1) mask with keep_prob = self.rate
        # and scale by 1/keep_prob (inverted dropout) to keep expected activations unchanged
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask elementwise
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Pass gradients only through the units that were kept in forward
        self.dinputs = dvalues * self.binary_mask