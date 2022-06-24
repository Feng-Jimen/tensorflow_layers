FROM tensorflow/tensorflow:latest

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

RUN pip install Pillow
RUN pip install opencv-python
RUN pip install matplotlib