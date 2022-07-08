import tensorflow as tf
from tensorflow.keras.layers import Input,StringLookup
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    vocab_data = ["a", "b", "c", "d"]
    
    inputs = Input(shape=(None,), dtype=tf.string)
    outputs = StringLookup(vocabulary=vocab_data)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    input_data = np.array([["a", "c", "d"], ["d", "z", "b"]])
    output_array = model(input_data)

    print(output_array)
    
    layer = StringLookup()
    layer.adapt([["a", "c", "d"], ["d", "z", "b"]])

    outputs2 = layer(inputs)
    
    model2 = Model(inputs=inputs,outputs=outputs2)

    model2.summary()
    
    output_array2 = model2(input_data)

    print(output_array2)
