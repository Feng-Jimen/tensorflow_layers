import tensorflow as tf

if __name__ == "__main__":
    print("items")
    
    for item in tf.keras.layers.__dict__.items():
        print(item)