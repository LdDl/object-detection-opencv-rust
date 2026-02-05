mkdir -p pretrained
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights -O pretrained/yolov4.weights
curl https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg --create-dirs -o pretrained/yolov4.cfg
# Make sure that batch size is for inference needs only
sed 's/batch=64/batch=1/g' -i pretrained/yolov4.cfg
sed 's/subdivision=8/subdivision=1/g' -i pretrained/yolov4.cfg
