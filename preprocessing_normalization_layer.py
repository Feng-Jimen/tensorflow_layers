import tensorflow as tf
from tensorflow.keras.layers import Input,Normalization
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
    input_data = np.array([1., 2., 3.], dtype='float32')

    layer = Normalization(axis=None)
    layer.adapt(adapt_data)
    
    inputs = Input(shape=(None,))    
    outputs = layer(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(input_data)

    print(output_array)
    
    print("NumPy calculate normalization")
    
    mean = adapt_data.mean(axis=None, keepdims=True)
    std  = np.std(adapt_data, axis=None, keepdims=True)
    score = (input_data-mean)/std
    
    print(score)
