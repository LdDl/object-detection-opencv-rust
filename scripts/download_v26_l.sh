# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt --create-dirs -o pretrained/yolo26l.pt
mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt -O pretrained/yolo26l.pt
printf "\n\n"
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "${RED}Make sure that you have installed 'ultralytics' for Python environment${NC}"
printf "\n\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolo26l.pt"); model.export(format="onnx", imgsz=640, simplify=True, opset=12, end2end=False)'
