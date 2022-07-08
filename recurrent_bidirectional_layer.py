import tensorflow as tf
from tensorflow.keras.layers import Input,LSTM,Bidirectional
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
       
    y = y.reshape([100,1, 1])
    
    inputs = Input(shape=(1,1))
    bidirectional = Bidirectional(LSTM(1, return_sequences=True))(inputs)
    outputs = Bidirectional(LSTM(1))(bidirectional)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(y)
    
    output_data1=output_array[:,0].numpy()
    output_data2=output_array[:,1].numpy()

    plt.plot(x,output_data1,label="output1")
    plt.plot(x,output_data2,label="output2")
    plt.legend()
    fig.savefig('result_recurrent_bidirectional.png')
