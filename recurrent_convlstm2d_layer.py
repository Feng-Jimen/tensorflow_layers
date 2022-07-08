import tensorflow as tf
from tensorflow.keras.layers import Input,ConvLSTM2D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    im = cv2.imread('sample_picture.jpg')
    im = im[None,...]
    
    inputs = Input(shape=im.shape)
    outputs = ConvLSTM2D(2,3,activation='linear')(inputs)
    
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
    
    filter_1=output_im[0,:,:,0].numpy()
    filter_2=output_im[0,:,:,1].numpy()
    
    cv2.imwrite('result_recurrent_convlstm2d_filter_1.jpg', filter_1)
    cv2.imwrite('result_recurrent_convlstm2d_filter_2.jpg', filter_2)
