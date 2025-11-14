class FullyConected:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(1. / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.X = None
    def forward(self, X):
        self.X = X  # (batch, in_features)
        return X.dot(self.W) + self.b  # (batch, out_features)
    def backprop(self, dY, lr):
        # dY: (batch, out_features)
        dW = self.X.T.dot(dY)
        db = np.sum(dY, axis=0, keepdims=True)
        dX = dY.dot(self.W.T)
        self.W -= lr * dW
        self.b -= lr * db
        return dX