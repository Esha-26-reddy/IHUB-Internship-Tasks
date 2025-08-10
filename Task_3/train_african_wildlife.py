from ultralytics import YOLO

# Load the base YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt depending on your use

# Train the model
model.train(
    data="D:/IIIT-H internship/african-wildlife/data.yaml",
    epochs=10,
    imgsz=640
)
