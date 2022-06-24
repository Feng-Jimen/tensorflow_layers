import tensorflow as tf
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras import Model
import numpy as np

def lambda_sample(x):
    print("enter lambda_sample function")
    print(x)
    x=x+1
    print("end lambda_sample function")
    return x
    
if __name__ == "__main__":
    inputs = Input(shape=(1))
    outputs = Lambda(lambda_sample)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()

    input_array = np.array([0,1,2,3,4,5])
    print(input_array)

    output_array = model(input_array)
    print(output_array)
