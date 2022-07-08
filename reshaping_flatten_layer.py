import tensorflow as tf
from tensorflow.keras.layers import Input,Flatten
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    freq=1
    x=np.linspace(0,3,100)
    input_data=np.sin(2*np.pi*x*freq)

    input_data = input_data.reshape([100, 1])
    
    inputs = Input(shape=input_data.shape)
    outputs = Flatten()(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    input_data =input_data[None,...]
    
    output_array = model(input_data)
    output = output_array[0]
    
    print(output)
        