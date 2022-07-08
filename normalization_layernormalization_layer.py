import tensorflow as tf
from tensorflow.keras.layers import Input,LayerNormalization
from tensorflow.keras import Model
import numpy as np

if __name__ == "__main__":
    adapt_data = np.array([[1.,1.,1.,1.,1.], [2.,2.,2.,2.,2.], [3.,3.,3.,3.,3.], [4.,4.,4.,4.,4.], [5.,5.,5.,5.,5.]], dtype='float32')
    input_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
    input_data = input_data[None,...]
 
    # inputs = Input(shape=(height,width,channel))
    inputs = Input(shape=(5,))
    outputs = LayerNormalization()(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    print(f'adapting data is {adapt_data}')
    for i in range(1000):
        output_im = model(adapt_data, training=True)
    
    print(f'adapting end')
    print(output_im)
    
    for w in model.non_trainable_weights:
        print(w)
        
    output_im = model(input_data, training=False)
    
    print(f'prediction {input_data}')
    print(output_im)
    
    adapt_data2 = np.array([[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]], dtype='float32')
    print(f'adapting data is {adapt_data2}')
    
    model2 = Model(inputs=inputs,outputs=outputs)
    for i in range(1000):
        output_im = model2(adapt_data2, training=True)
    
    print(f'adapting end')
    print(output_im)
    
    for w in model2.non_trainable_weights:
        print(w)
        
    output_im = model(input_data, training=False)
    
    print(f'prediction {input_data}')
    print(output_im)

    
    print("NumPy calculate normalization")
    
    mean = adapt_data.mean(axis=None, keepdims=True)
    std  = np.std(adapt_data, axis=None, keepdims=True)
    score = (input_data-mean)/std
    
    print(score)
