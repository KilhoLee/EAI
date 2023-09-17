## APP_yolo_with_pycam

**prerequisite : compile raspicam [C++ API for using RPI pycam]**

git clone https://github.com/cedricve/raspicam.git  && compile raspicam 


**yolo_with_pycam flow**

1. load image from pycam & load model (yolov4-tiny)
2. Inference on TFLite
3. Do output parsing 
4. Visualize

*should use this project on RPI, connected with Pycam*
