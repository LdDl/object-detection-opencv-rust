#!/bin/bash
# Download ArcFace ResNet50 (w600k_r50) from InsightFace model zoo
# Model: ResNet50 trained on WebFace600K, 512-dim embedding
# License: MIT
# Size: ~166 MB
# Note: Use ArcFaceNorm::ResNet for this model (pixel / 255.0)

mkdir -p pretrained

wget https://huggingface.co/WePrompt/buffalo_l/resolve/main/w600k_r50.onnx -O pretrained/w600k_r50.onnx

printf "\n"
echo "Downloaded: pretrained/w600k_r50.onnx"
echo "Use with: Model::arcface_ort_with_norm(\"pretrained/w600k_r50.onnx\", ArcFaceNorm::ResNet)"

# Also download YuNet detector if not present
if [ ! -f pretrained/face_detection_yunet_2023mar.onnx ]; then
    echo ""
    echo "YuNet detector not found, downloading..."
    wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
        -O pretrained/face_detection_yunet_2023mar.onnx
    echo "Downloaded: pretrained/face_detection_yunet_2023mar.onnx"
fi
