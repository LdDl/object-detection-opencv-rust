# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt --create-dirs -o pretrained/yolo11l.pt
mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -O pretrained/yolo11l.pt
printf "\n\n"
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "${RED}Make sure that you have installed 'ultralytics' for Python environment${NC}"
printf "\n\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolo11l.pt"); model.export(format="onnx", imgsz=640, simplify=True, opset=12)'
