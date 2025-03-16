# Jockey Detection and Tracking for horse racing using YOLO v12
## Overview
This project attempts to detect and track jockeys during horse racing. Object detection framework Yolo v12 is used to detect and track jockeys.

## Training
Download training  data
$ curl -L "https://universe.roboflow.com/ds/FrDgZN82Rn?key=3kcd17C5db" > roboflow.zip
$ mkdir datasets
$ cd datasets
$ unzip roboflow.zip
$ rm roboflow.zip

Train a detection model based on yolo12l.pt
$ yolo detect train data=datasets/data.yaml model=yolo12l.pt epochs=120 imgsz=640 batch=6
