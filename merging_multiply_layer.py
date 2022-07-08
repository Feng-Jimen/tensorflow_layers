import tensorflow as tf
from tensorflow.keras.layers import Input,Multiply
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    # input_array = [[0,1,2,3,4,5]]
    input_array1 = np.array([[0,1,2,3,4,5],[100,101,102,103,104,105]])
    print(input_array1)
    input_array2 = np.array([[10,11,12,13,14,15],[20,21,22,23,24,25]])
    print(input_array2)

    inputs1 = Input(shape=(2,6))
    inputs2 = Input(shape=(2,6))
    outputs = Multiply()([inputs1,inputs2])
    
    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
    
    model.summary()
    
    input_array1 = input_array1[None,...]
    input_array2 = input_array2[None,...]

    output_array = model([input_array1,input_array2])
    print(output_array)
