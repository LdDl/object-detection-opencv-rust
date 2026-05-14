#!/bin/bash
# Download ArcFace MobileFaceNet (w600k_mbf) from InsightFace model zoo
# Model: MobileFaceNet trained on WebFace600K, 512-dim embedding
# License: MIT
# Size: ~14 MB

mkdir -p pretrained

# w600k_mbf.onnx from InsightFace (buffalo_sc pack)
# Original source was github.com/deepinsight/insightface/releases/download/v0.7/ (now 404)
# Mirror: HuggingFace
wget https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx -O pretrained/w600k_mbf.onnx

printf "\n"
echo "Downloaded: pretrained/w600k_mbf.onnx"
echo "Use with: Model::arcface_ort(\"pretrained/w600k_mbf.onnx\")"

# Also download YuNet detector if not present
if [ ! -f pretrained/face_detection_yunet_2023mar.onnx ]; then
    echo ""
    echo "YuNet detector not found, downloading..."
    wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
        -O pretrained/face_detection_yunet_2023mar.onnx
    echo "Downloaded: pretrained/face_detection_yunet_2023mar.onnx"
fi
