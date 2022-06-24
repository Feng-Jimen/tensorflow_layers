import tensorflow as tf
from tensorflow.keras.layers import Input,AvgPool1D
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
   
    y = y.reshape([100, 1])
    
    inputs = Input(shape=y.shape)
    outputs = AvgPool1D(pool_size=2,strides=1, padding='same')(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y])
    output=output_array[0,:,0].numpy()
    plt.plot(x,output,label="output")
    plt.legend()
    fig.savefig('result_pooling_avgpool1d.png')
