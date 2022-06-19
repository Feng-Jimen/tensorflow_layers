import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Input,InputSpec
from tensorflow.keras import Model, Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.input_spec = InputSpec(shape=(32,32,3))
        self.inputs = InputLayer(input_shape=(32,32,3))

    def call(self,x):
        return self.inputs(x)

if __name__ == "__main__":
    print("sequential model")
    model_sequential = Sequential([
            InputLayer(input_shape=(32,32,3)), 
        ])

    model_sequential.summary()
    
    print("functional model")
    inputs = Input(shape=(32,32,3))
    
    model_functional = Model(inputs=inputs,outputs=inputs)
    
    model_functional.summary()
    
    print("subclassing model")
    model_subclass = MyModel()
    model_subclass.build(input_shape=(32,32,3))
    
    model_subclass.summary()