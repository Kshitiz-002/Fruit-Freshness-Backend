from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

# Display model information (optional)
model.info()

results = model.train(data="C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/config.yaml", epochs=1)
