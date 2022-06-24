import tensorflow as tf
from tensorflow.keras.layers import Input,GlobalAvgPool1D
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    y = np.array([1,2,3,4,5,6])
    
    y = y.reshape([6, 1])
    
    inputs = Input(shape=y.shape)
    outputs = GlobalAvgPool1D()(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y])
    
    output = output_array[0][0].numpy()
    
    print(f'input avg:{np.average(y[0,:,0])}')
    print(f'output {output}')
