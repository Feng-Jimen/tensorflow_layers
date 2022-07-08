import tensorflow as tf
from tensorflow.keras.layers import Input,Activation
from tensorflow.keras import Model
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def test_activation_layer(activation_name):
    x=np.linspace(-3,3,400)
    
    fig = plt.figure()
    
    fig.suptitle(activation_name)
    y = x.reshape([400, 1])
    
    inputs = Input(shape=y.shape)
    outputs = Activation(activation_name)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)

    model.summary()
        
    output_array = model(y)
    output_array = output_array[:,0,0]
    
    plt.plot(x,output_array,label="output")
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid(color='b', linestyle=':', linewidth=0.3)

    fig.savefig('result_activation_activation_' + activation_name + '.png')
    plt.clf()


if __name__ == "__main__":
    mpl.use('Agg')
    x=np.linspace(-2,2,400)
    
    test_activation_layer('elu')
    test_activation_layer('exponential')
    test_activation_layer('gelu')
    test_activation_layer('hard_sigmoid')
    test_activation_layer('linear')
    test_activation_layer('relu')
    test_activation_layer('selu')
    test_activation_layer('sigmoid')
    test_activation_layer('softmax')
    test_activation_layer('softplus')
    test_activation_layer('softsign')
    test_activation_layer('swish')
    test_activation_layer('tanh')
