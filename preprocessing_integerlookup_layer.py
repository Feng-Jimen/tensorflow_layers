import tensorflow as tf
from tensorflow.keras.layers import Input,IntegerLookup
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    vocab = [12, 36, 1138, 42]
    
    inputs = Input(shape=(None,))
    outputs = IntegerLookup(vocabulary=vocab)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    input_data = np.array([[12, 1138, 42], [42, 1000, 36]])
    output_array = model(input_data)

    print(output_array)
    
    layer = IntegerLookup()
    layer.adapt([[12, 1138, 42], [42, 1000, 36]])

    outputs2 = layer(inputs)
    
    model2 = Model(inputs=inputs,outputs=outputs2)

    model2.summary()
    
    output_array2 = model2(input_data)

    print(output_array2)
