#!/bin/bash
# YOLOv5u medium - Ultralytics updated format (same output as YOLOv8)
# Use with: Model::ort("pretrained/yolov5mu.onnx", ...)

set -e

mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov5mu.pt -O pretrained/yolov5mu.pt

printf "\n\n"
RED='\033[0;31m'
NC='\033[0m'
printf "${RED}Installing dependencies...${NC}\n"
pip install ultralytics onnx

printf "${RED}Exporting to ONNX...${NC}\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolov5mu.pt"); model.export(format="onnx", opset=12)'

printf "\n${RED}Done! Model exported to pretrained/yolov5mu.onnx${NC}\n"
