import tensorflow as tf
from tensorflow.keras.layers import Input,UpSampling3D 
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
            frame_list.append(frame)
            num=num+1;
        else :
            break
    cap.release()
    
    input_array = np.array(frame_list)
    
    inputs = Input(shape=input_array.shape)
    outputs = UpSampling3D(size=(2, 2, 2))(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
   
    model.summary()
    
    input_array = input_array[None,...]
    
    
    output_im = model(input_array)

    output_im = tf.cast(output_im, tf.uint8)
    
    output_im=output_im[0,:,:,:,:].numpy()
    
    output_frames, output_height,output_width,output_channel = output_im.shape
    
    print(output_im.shape)
    
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    writer = cv2.VideoWriter('result_reshaping_upsampling3d.mp4', fmt, fps, (output_width, output_height))
    for n in range(output_frames):
        bgr = output_im[n]
        writer.write(bgr)
    writer.release()

    cv2.destroyAllWindows()
