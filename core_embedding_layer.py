import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    inputs = Input(shape=(1))
    outputs = Embedding(6,2)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.summary()

    # input_array = [[0,1,2,3,4,5]]
    input_array = np.array([0,1,2,3,4,5])
    print(input_array)

    # output_array = model.predict(input_array)
    output_array = model(input_array)
    print(output_array)
