import tensorflow as tf
from tensorflow.keras.layers import Input,RandomBrightness
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    height,width,channel = im.shape
    
    inputs = Input(shape=im.shape)
    outputs = RandomBrightness(factor=(-0.9, 0.9))(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array)
    
    output=output_im[0,:,:,:].numpy()
    
    cv2.imwrite('result_preprocessing_randombrightness_output.jpg', output)
