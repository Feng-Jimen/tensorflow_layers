import tensorflow as tf
from tensorflow.keras.layers import Input,RNN,LSTMCell
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
    # outputs = LSTM(1, return_sequences=True, return_state=True)(inputs)
    outputs = RNN(LSTMCell(1), return_state=True)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
    
    output_array = model(y)
    
    # output_data=output_array[0][:,0,0].numpy()
    output_data=output_array[0][:,0].numpy()
    # output_sequences=output_array[1].numpy()
    output_state=output_array[2].numpy()

    plt.plot(x,output_data,label="output")
    # plt.plot(x,output_sequences,label="sequences")
    plt.plot(x,output_state,label="state")
    plt.legend()
    fig.savefig('result_recurrent_lstmcell.png')

