import tensorflow as tf

if __name__ == "__main__":
    print("activations")
    
    for item in tf.keras.activations.__dict__.items():
        print(item)