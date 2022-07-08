import tensorflow as tf
from tensorflow.keras.layers import Input,TextVectorization
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    vocab_data = ["earth", "wind", "and", "fire"]
    
    inputs = Input(shape=(1,), dtype=tf.string)
    outputs = TextVectorization(max_tokens=5000,output_mode='int',output_sequence_length=1,vocabulary=vocab_data)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    input_data = np.array(["earth", "wind", "and", "fire"])
    output_array = model(input_data)

    print(output_array)
    
    input_data = np.array(["Earth,", "Wind", "&", "Fire"])
    output_array = model(input_data)

    print(output_array)