import tensorflow as tf
from tensorflow.keras.layers import Input,UpSampling1D
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mpl.use('Agg')
    freq=1
    x=np.linspace(0,3,100)
    x2=np.linspace(0,3,200)
    y=np.sin(2*np.pi*x*freq)
    
    fig = plt.figure()
       
    y = y.reshape([100, 1])
    
    inputs = Input(shape=y.shape)
    outputs = UpSampling1D(size=2)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y])

    output=output_array[0,:,0].numpy()

    plt.plot(x2,output,label="output")
    plt.legend()
    fig.savefig('result_reshaping_upsampling1d.png')
