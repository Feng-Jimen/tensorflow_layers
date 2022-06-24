import tensorflow as tf
from tensorflow.keras.layers import Input,GlobalMaxPool2D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    
    inputs = Input(shape=im.shape)
    outputs = GlobalMaxPool2D()(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array)
    
    input_r_max = max(np.ravel(im[0,:,:,0]))
    input_g_max = max(np.ravel(im[0,:,:,1]))
    input_b_max = max(np.ravel(im[0,:,:,2]))
    print(f'Input Max RGB:{input_r_max} {input_g_max} {input_b_max}')
    
    r_max = output_im[0,0].numpy()
    g_max = output_im[0,1].numpy()
    b_max = output_im[0,2].numpy()
    print(f'Output RGB:{r_max} {g_max} {b_max}')
