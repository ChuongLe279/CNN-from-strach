import numpy as np

class maxPooling: 

    def __init__(self, kernel_size, stride): 
        # Initialize kernel size and stride 
        self.kernel_size = kernel_size 
        self.stride = stride

    def sliding_window(self, image):
        # Compute the ouput size
        new_height = (image.shape[1] - self.kernel_size) // self.stride + 1 
        new_width = (image.shape[2] - self.kernel_size) // self.stride + 1 

        # Store for backprop 
        self.last_image = image 
         
        for i in range(new_height): 
            for j in range(new_width): 
                patch = image[:, (i*self.stride):(i*self.stride + self.kernel_size), 
                              (j*self.stride):(j*self.stride + self.kernel_size)]
                yield patch, i, j

    def forward(self, image): 

        num_filters, height, width = image.shape 
        output = np.zeros((num_filters, (height - self.kernel_size)//self.stride + 1, (width - self.kernel_size)//self.stride + 1))

        for patch, i, j in self.sliding_window(image): 
            output[:, i, j] = np.amax(patch, axis=(1, 2))

        return output
    
    def backprop(self, dE_dY): 
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically softmax.
        There are no weights to update, but the output is needed to update the weights of the convolutional layer.
        """
        dE_dk = np.zeros(self.last_image.shape)
        for patch, i, j in self.sliding_window(self.last_image): 
            h, w, f = patch.shape 
            amax = np.amax(patch, axis=(0, 1))

            for idx_h in range(h): 
                for idx_w in range(w): 
                    for idx_f in range(f): 
                        if patch[idx_h, idx_w, idx_f] == amax[idx_f]: 
                            dE_dk[i*self.stride + idx_h, j*self.stride + idx_w, idx_f] = dE_dY[i, j, idx_f]
        return dE_dk