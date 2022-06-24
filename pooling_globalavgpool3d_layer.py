import tensorflow as tf
from tensorflow.keras.layers import Input,GlobalAvgPool3D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("sample_video.mp4")
    
    frame_list=[]

    while(1):
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else :
            break
    cap.release()
    
    input_array = np.array(frame_list)
    
    inputs = Input(shape=input_array.shape)
    outputs = GlobalAvgPool3D()(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    input_array = input_array[None,...]
    
    output_im = model(input_array)
    
    input_r_max = np.average(np.ravel(input_array[0,:,:,:,0]))
    input_g_max = np.average(np.ravel(input_array[0,:,:,:,1]))
    input_b_max = np.average(np.ravel(input_array[0,:,:,:,2]))
    print(f'Input Avg RGB:{input_r_max} {input_g_max} {input_b_max}')
    
    r_max = output_im[0,0].numpy()
    g_max = output_im[0,1].numpy()
    b_max = output_im[0,2].numpy()
    print(f'Output RGB:{r_max} {g_max} {b_max}')
