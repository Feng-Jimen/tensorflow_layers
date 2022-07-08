import tensorflow as tf
from tensorflow.keras.layers import Input,GaussianDropout
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
    outputs = GaussianDropout(.2)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y], training=True)
    
    output = output_array[0]

    plt.plot(x,output,label="output")
    plt.legend()
    fig.savefig('result_regularization_gaussiandropout.png')
