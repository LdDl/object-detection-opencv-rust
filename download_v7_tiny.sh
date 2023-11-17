curl https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights --create-dirs -o pretrained/yolov7-tiny.weights
curl https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg --create-dirs -o pretrained/yolov7-tiny.cfg
# Make sure that batch size is for inference needs only
sed 's/batch=64/batch=1/g' -i pretrained/yolov7-tiny.cfg