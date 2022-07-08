import tensorflow as tf
from tensorflow.keras.layers import Input,Hashing 
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    input_data = np.array([['A'], ['B'], ['C'], ['D'], ['E']])

    inputs = Input(shape=(), dtype=tf.string)    
    outputs = Hashing(num_bins=5)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(input_data)

    print(output_array)
    
