import tensorflow as tf
from tensorflow.keras.layers import Input,Conv3D
from tensorflow.keras import Model
import cv2
import numpy as np

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("sample_video.mp4")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    num = 0
    frame_list=[]
    while(1):
        ret, frame = cap.read()
        if ret:
            frame = frame / 255
            frame_list.append(frame)
            num=num+1;
        else :
            break
    cap.release()
    
    input_array = np.array(frame_list)
    
    inputs = Input(shape=input_array.shape)
    outputs = Conv3D(2,3,activation=None,data_format='channels_last')(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    input_array = input_array[None,...]
    
    
    output_im = model(input_array)
    
    max_value = max(np.ravel(output_im))
    min_value = min(np.ravel(output_im))
    output_im = ((output_im - min_value) / (max_value - min_value)) * 255
    output_im = tf.cast(output_im, tf.uint8)
    
    filter_1=output_im[0,:,:,:,0].numpy()
    filter_2=output_im[0,:,:,:,1].numpy()
    
    filter_1_frames, filter_1_height, filter_1_width = filter_1.shape
    filter_2_frames, filter_2_height, filter_2_width = filter_2.shape
    
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    writer = cv2.VideoWriter('result_convolution_conv3d_filter_1.mp4', fmt, fps, (filter_1_width, filter_1_height))
    for n in range(filter_1_frames):
        bgr = cv2.cvtColor(filter_1[n], cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()
    
    writer = cv2.VideoWriter('result_convolution_conv3d_filter_2.mp4', fmt, fps, (filter_2_width, filter_2_height))
    for n in range(filter_2_frames):
        bgr = cv2.cvtColor(filter_2[n], cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()

    cv2.destroyAllWindows()
