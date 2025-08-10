from ultralytics import YOLO
model = YOLO('yolov8n.pt') 
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)
for result in results:
    print(result.names) 
    print(result.boxes)  
