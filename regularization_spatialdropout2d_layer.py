import tensorflow as tf
from tensorflow.keras.layers import Input,SpatialDropout2D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    
    inputs = Input(shape=im.shape)
    outputs = SpatialDropout2D(.5)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array, training=True)
    
    print(output_im)
    
    output=output_im[0,:,:,:].numpy()
    
    cv2.imwrite('result_regularization_spatialdropout2d.jpg', output)
