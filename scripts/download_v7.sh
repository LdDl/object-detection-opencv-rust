mkdir -p pretrained
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights -O pretrained/yolov7.weights
curl https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7.cfg --create-dirs -o pretrained/yolov7.cfg
# Make sure that batch size is for inference needs only
sed 's/batch=8/batch=1/g' -i pretrained/yolov7.cfg
