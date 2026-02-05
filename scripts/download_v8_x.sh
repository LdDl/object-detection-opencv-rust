# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt --create-dirs -o pretrained/yolov8x.pt
mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O pretrained/yolov8x.pt
printf "\n\n"
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "${RED}Make sure that you have installed 'ultralytics' for Python environment${NC}"
printf "\n\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolov8x.pt"); model.export(format="onnx", opset=12)'