from ultralytics import YOLO

# Step 1: Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Step 2: Run inference on a single image or a folder
results = model("test_images/", save=True, conf=0.25)  # <- update path if needed

# Done: Results saved to runs/detect/predict
