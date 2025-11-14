import numpy as np

def Convolution_Function(image, kernels,bias=None, stride=1, padding =0,activation=True):
    # lay kich thuoc input va kernel
    c,h,w=image.shape
    filters,ck,hk,wk=kernels.shape

    # kiem tra so kenh hop le
    assert c == ck, "Error"

    #them padding (0)
    if padding > 0:
        input_image = np.pad(input_image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    #Tinh kich thuoc dau ra
    out_h=int((h-hk+2*padding)/stride)+1
    out_w=int((w-wk+2*padding)/stride)+1

    #Khoi tao gia tri dau ra
    output=np.zeros((filters,out_h, out_w))

    #tich chap tung filter
    for f in range(filters):
        for i in range(out_h):
            for j in range(out_w):
                region=image[:,i*stride : i*stride + hk, j*stride : j*stride +wk]
                output[f,i,j]=np.sum(region* kernels[f])
        if bias is not None:
            output[f]+=bias[f]
    if activation:
        ouput=np.maximum(0,output)
    return output
    

    

    