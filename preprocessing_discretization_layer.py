import tensorflow as tf
from tensorflow.keras.layers import Input,Discretization
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    input_data = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])

    inputs = Input(shape=(None,))
    outputs = Discretization(bin_boundaries=[0., 1., 2.])(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(input_data)

    print(output_array)
    
    layer = Discretization(num_bins=4, epsilon=0.01)
    layer.adapt(input_data)
    
    outputs2 = layer(inputs)

    model2 = Model(inputs=inputs,outputs=outputs2)

    model2.summary()
    
    output_array2 = model2(input_data)

    print(output_array2)
