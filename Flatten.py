class Flatten:
    def __init__(self): self.shape = None
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0] * X.shape[1] * X.shape[2], 1)
    def backward(self, dY):
        return dY.reshape(self.shape)