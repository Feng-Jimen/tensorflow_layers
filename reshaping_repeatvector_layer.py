import tensorflow as tf
from tensorflow.keras.layers import Input,RepeatVector
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    inputs = Input(shape=(6))
    outputs = RepeatVector(3)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()

    input_array = np.array([0,1,2,3,4,5])
    print(input_array)
    input_array=input_array[None,...]

    output_array = model(input_array)
    output_array=output_array[0]
    print(output_array)

        