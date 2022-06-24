import tensorflow as tf
from tensorflow.keras.layers import Input,Masking
from tensorflow.keras import Model, Sequential
import numpy as np

if __name__ == "__main__":
    inputs = Input(shape=(1))
    outputs = Masking(mask_value=4)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()

    input_array = np.array([0,1,2,3,4,5])
    print(input_array)

    output_array = model(input_array)
    print(output_array)

    print("_keras_mask")
    print(output_array._keras_mask)

