import tensorflow as tf
from tensorflow.keras.layers import Input,CategoryEncoding 
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    input_data = np.array([3, 2, 0, 1])

    inputs = Input(shape=())    
    outputs = CategoryEncoding(num_tokens=4, output_mode="one_hot")(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(input_data)

    print(output_array)
    
