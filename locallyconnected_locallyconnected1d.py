import tensorflow as tf
from tensorflow.keras.layers import Input,LocallyConnected1D
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
    outputs = LocallyConnected1D(2,3,activation='linear',padding='valid')(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    y = y[None,...]

    output_array = model([y])

    filter_1=output_array[0,:,0].numpy()
    filter_2=output_array[0,:,1].numpy()
    
    output_array1 = np.array([0])
    output_array1 = np.append(output_array1, filter_1)
    output_array1 = np.append(output_array1, 0)

    output_array2 = np.array([0])
    output_array2 = np.append(output_array2, filter_2)
    output_array2 = np.append(output_array2, 0)

    plt.plot(x,output_array1,label="filter_1")
    plt.plot(x,output_array2,label="filter_2")
    plt.legend()
    fig.savefig('result_locallyconnected_locallyconnected1d.png')
