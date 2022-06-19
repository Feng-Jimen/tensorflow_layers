import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Input,InputSpec,Dense
from tensorflow.keras import Model, Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.input_spec = InputSpec(shape=(32,32,3))
        self.inputs = InputLayer(input_shape=(32,32,3))
        self.dense = Dense(128,activation='relu')

    def call(self,x):
        x = self.inputs(x)
        return self.dense(x)

if __name__ == "__main__":
    print("sequential model")
    model_sequential = Sequential([
            InputLayer(input_shape=(32,32,3)), 
            Dense(128,activation='relu'),
        ])

    model_sequential.summary()
    
    print("functional model")
    inputs = Input(shape=(32,32,3))
    outputs = Dense(128,activation='relu')(inputs)
    
    model_functional = Model(inputs=inputs,outputs=outputs)
    
    model_functional.summary()
    
    print("subclassing model")
    model_subclass = MyModel()
    model_subclass.build(input_shape=(32,32,3))
    
    model_subclass.summary()