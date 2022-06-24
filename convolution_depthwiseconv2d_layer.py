import tensorflow as tf
from tensorflow.keras.layers import Input,DepthwiseConv2D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')

    print(im.shape)
    
    inputs = Input(shape=im.shape)
    outputs = DepthwiseConv2D(3,activation='linear')(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    im = im / 255
    im = im[None,...]
    
    input_array = np.array(im)

    output_im = model(input_array)
    
    max_value = max(np.ravel(output_im))
    min_value = min(np.ravel(output_im))
    output_im = ((output_im - min_value) / (max_value - min_value)) * 255
    output_im = tf.cast(output_im, tf.int32)
    
    output=output_im[0,:,:,:].numpy()
    cv2.imwrite('result_convolution_depthwiseconv2d_output.jpg', output)
