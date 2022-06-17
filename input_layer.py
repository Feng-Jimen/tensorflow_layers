import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Input,Flatten
from tensorflow.keras import Model, Sequential

if __name__ == "__main__":
    model = Sequential([
            InputLayer(input_shape=(32,32,3)), 
            Flatten()
        ])

    model.summary()