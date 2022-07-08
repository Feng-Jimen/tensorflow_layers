import tensorflow as tf
from tensorflow.keras.layers import Input,SpatialDropout1D
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mpl.use('Agg')
    freq=1
    x=np.linspace(0,3,100)
    y=np.sin(2*np.pi*x*freq)
    
    fig = plt.figure()
    
    plt.plot(x,y,label="input")

    y2 = y.reshape([100, 1])
    y = y.reshape([100, 1, 1])

    inputs = Input(shape=(1,1))
    outputs = SpatialDropout1D(.2)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    # y = y[...,None]

    output_array = model(y, training=True)
    
    output = output_array[:,0,0]

    plt.plot(x,output,label="output")
    
    inputs2 = Input(shape=y2.shape)
    outputs2 = SpatialDropout1D(.2)(inputs2)
    
    model2 = Model(inputs=inputs2,outputs=outputs2)
    
    y2 = y2[None,...]
    
    output_array2 = model2(y2, training=True)
    
    output2 = output_array2[0]

    plt.plot(x,output2,label="output2")
    
    plt.legend()
    fig.savefig('result_regularization_spatialdropout1d.png')
