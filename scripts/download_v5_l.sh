#!/bin/bash
# Original YOLOv5 large - classic format with objectness score
# Use with: Model::yolov5_ort("pretrained/yolov5l.onnx", ...)

set -e

mkdir -p pretrained
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt -O pretrained/yolov5l.pt

printf "\n\n"
RED='\033[0;31m'
NC='\033[0m'
printf "${RED}Cloning yolov5 repo for export script...${NC}\n"

# Clone yolov5 repo if not exists
if [ ! -d "yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git --depth 1
fi

# Install requirements
pip install -r yolov5/requirements.txt onnx

# Export to ONNX (creates file next to .pt)
python yolov5/export.py --weights pretrained/yolov5l.pt --include onnx --opset 12

printf "\n${RED}Done! Model exported to pretrained/yolov5l.onnx${NC}\n"
