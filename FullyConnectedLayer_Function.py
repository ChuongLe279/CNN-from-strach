import numpy as np

def fully_connected(input_vector, weights, bias, activation=True):
   
    # Nhân ma trận W.x + b
    z = np.dot(weights, input_vector) + bias

    # Hàm kích hoạt (ReLU)
    if activation:
        z = np.maximum(0, z)
    return z
