# Jockey Detection and Tracking for horse racing using YOLO v12
## Overview
This project attempts to detect and track jockeys during horse racing. Object detection framework Yolo v12 is used to detect and track jockeys.

## Training
Download training data
<pre>
$ curl -L "https://universe.roboflow.com/ds/FrDgZN82Rn?key=3kcd17C5db" > roboflow.zip
$ mkdir datasets
$ cd datasets
$ unzip roboflow.zip
$ rm roboflow.zip
</pre>
Train a detection model based on yolo12l.pt
<pre>$ yolo detect train data=datasets/data.yaml model=yolo12l.pt epochs=120 imgsz=640 batch=6</pre>

## Inference
Detect and track jockeys
<pre>$ python predict.py --source_weights_path yolo12l.pt --source_video_path racing_video.mp4 --target_video_path output_racing.mp4 </pre>
