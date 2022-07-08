import tensorflow as tf
from tensorflow.keras.layers import Input,Cropping2D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    
    inputs = Input(shape=im.shape)
    outputs = Cropping2D(cropping=((10, 10), (20, 20)))(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array)

    output_im=output_im[0,:,:,:].numpy()
    
    cv2.imwrite('result_reshaping_cropping2d_output.jpg', output_im)
