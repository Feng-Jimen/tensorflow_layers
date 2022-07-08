import tensorflow as tf
from tensorflow.keras.layers import Input,Attention,Dense
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mpl.use('Agg')
    freq=1
    x=np.linspace(0,3,100)
    query_ori=np.sin(2*np.pi*x*freq)
    value_ori=np.cos(2*np.pi*x*freq)
    # value=x/3 + 0.5
    
    fig = plt.figure()
    
    plt.plot(x,query_ori,label="query")
    plt.plot(x,value_ori,label="value")
   
    query = query_ori.reshape([100, 1])
    value = value_ori.reshape([100, 1])
    
    query_input = Input(shape=query.shape)
    value_input  = Input(shape=value.shape)
    outputs,scores = Attention()([query_input, value_input], return_attention_scores=True)
    
    model = Model(inputs=[query_input, value_input],outputs=[outputs,scores])

    model.summary()
    
    query = query_ori[None,...]
    value = value_ori[None,...]

    output_array,attention_scores = model([query, value])
    # output = output_array[:,0,0]
    output = output_array[0]
        
    attention_scores = attention_scores[0]
    matmul_value = np.matmul(attention_scores, value_ori)
    
    plt.plot(x,output,label="output")
    plt.plot(x,matmul_value,label="matmul")
    plt.legend()
    fig.savefig('result_attention_attention.png')
