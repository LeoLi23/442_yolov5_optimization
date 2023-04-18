from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
path = './runs/train/exp3/weights/best.pt'
model = YOLO(path)  # load a custom model

# Predict with the model
source = './dataset/UOT_dataset/images/test/1.jpg'
model.predict(source, save=True, show=True, imgsz=640, conf=0.5)
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image