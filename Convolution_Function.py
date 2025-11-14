import numpy as np

class Convolution:
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        # in_channels: số kênh đầu vào, num_filters: số filter (số kênh đàu ra)
        self.in_channels = input_channels
        self.out_channels = num_filters
        # Đảm bảo kernel_size là tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Khởi tạo trọng số và bias (ngẫu nhiên hoặc zeros)
        # self.weights là list các kernel 
        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.bias = np.zeros(self.out_channels)
        self.X_padded = None
        
    def forward(self, image):
        # lay kich thuoc input va kernel
        c, h, w = image.shape
        hk, wk = self.kernels_size
        filters = self.out_channels

        # kiem tra so kenh hop le
        assert c == self.out_channels, "Error"

        #them padding (0)
        p = self.padding
        s = self.s
        if p > 0:
            input_image = np.pad(input_image, ((0, 0), (p, p), (p, p)), mode='constant')
        
        #Tinh kich thuoc dau ra
        out_h = int((h - hk + 2*p) / s) + 1
        out_w = int((w - wk + 2*p) / s) + 1

        #Khoi tao gia tri dau ra
        output = np.zeros((filters,out_h, out_w))

        #tich chap tung filter
        for f in range(filters):
                for i in range(out_h):
                    for j in range(out_w):
                        region = image[:,i*s : i*s + hk, j*s : j*s + wk]
                        output[f,i,j] = np.sum(region * self.weights[f]) + self.bias[f]
        return output
    
    def backprop(self, dY, learning_rate):
        c, out_h, out_w = dY.shape
        dX = np.zeros_like(self.X_padded)
        db = np.zeros_like(self.bias)
        dW = np.zeros_like(self.weights)
        s = self.stride
        hk, wk = self.kernel_size

        # Gradient descent
        for i in range(out_h):
            for j in range(out_w):
                region = self.X_padded[:, i*s : i*s + hk, j*s : j*s + wk]
                for f in range(self.out_channels):
                    dW[f] += np.sum(region * (dY[f, i, j]), axis=0)
                    dX[:, i*s : i*s + hk, j*s : j*s + wk] += self.weights[f] * dY[f, i, j]
                    db[f] += dY[f, i, j]

        # Loại bỏ lớp padding trong dX
        p = self.padding
        if p > 0:
            dX = dX[:, p : -p, p : -p]

        # Cập nhập trọng số
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db.reshape(self.bias.shape)
        return dX                
class ReLU:
    def __init__(self):
        self.cache = None
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)
    def backward(self, dY):
        X = self.cache
        dX = dY.copy()
        dX[X <= 0] = 0
        return dX