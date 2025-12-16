# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt --create-dirs -o pretrained/yolov9t.pt
mkdir -p pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt -O pretrained/yolov9t.pt
printf "\n\n"
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "${RED}Make sure that you have installed 'ultralytics' for Python environment${NC}"
printf "\n\n"
python3 -c 'from ultralytics import YOLO; model = YOLO("pretrained/yolov9t.pt"); model.export(format="onnx", imgsz=640, simplify=True, opset=12)'