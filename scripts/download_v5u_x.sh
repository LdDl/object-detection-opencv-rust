#!/bin/bash
# YOLOv5u extra - Ultralytics updated format (same output as YOLOv8)
# Use with: Model::ort("pretrained/yolov5xu.onnx", ...)

set -e

mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov5xu.pt -O pretrained/yolov5xu.pt

printf "\n\n"
RED='\033[0;31m'
NC='\033[0m'
printf "${RED}Installing dependencies...${NC}\n"
pip install ultralytics onnx

printf "${RED}Exporting to ONNX...${NC}\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolov5xu.pt"); model.export(format="onnx", opset=12)'

printf "\n${RED}Done! Model exported to pretrained/yolov5xu.onnx${NC}\n"
