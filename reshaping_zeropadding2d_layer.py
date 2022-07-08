import tensorflow as tf
from tensorflow.keras.layers import Input,ZeroPadding2D 
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    
    inputs = Input(shape=im.shape)
    outputs = ZeroPadding2D(padding=(10,20))(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array)

    output_im=output_im[0,:,:,:].numpy()
    
    cv2.imwrite('result_reshaping_zeropadding2d_output.jpg', output_im)
