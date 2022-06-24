import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D
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
    outputs = Conv1D(2,3,activation='linear',padding='same')(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y])

    filter_1=output_array[0,:,0].numpy()
    filter_2=output_array[0,:,1].numpy()

    plt.plot(x,filter_1,label="filter_1")
    plt.plot(x,filter_2,label="filter_2")
    plt.legend()
    fig.savefig('result_convolution_conv1d.png')
