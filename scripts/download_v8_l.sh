# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt --create-dirs -o pretrained/yolov8l.pt
mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt -O pretrained/yolov8l.pt
printf "\n\n"
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "${RED}Make sure that you have installed 'ultralytics' for Python environment${NC}"
printf "\n\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolov8l.pt"); model.export(format="onnx", opset=12)'