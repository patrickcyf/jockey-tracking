# Jockey Detection and Tracking for horse racing using YOLO v12
## Overview
This project attempts to detect and track jockeys during horse racing. Object detection framework Yolo v12 is used to detect and track jockeys.

## Training
Download training data
<pre>
$ cd datasets
$ curl -L "https://app.roboflow.com/ds/vmUZYhZXDx?key=CkMIyTH1xT" > roboflow.zip
$ unzip roboflow.zip
$ rm roboflow.zip
</pre>
Train a detection model based on yolo12l.pt
<pre>$ yolo detect train data=datasets/data.yaml model=yolo12l.pt epochs=120 imgsz=640 batch=6</pre>

## Inference
Detect and track jockeys
<pre>$ python predict.py --source_weights_path yolo12l.pt --source_video_path racing_video.mp4 --target_video_path output_racing.mp4 </pre>
