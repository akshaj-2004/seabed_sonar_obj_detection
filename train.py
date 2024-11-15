from ultralytics import YOLO  # Import the YOLO class from the Ultralytics YOLO library

model = YOLO("yolo11s.pt")  # Load a YOLO model with the specified weights file ("yolo11n.pt")

# Start training with the specified parameters
model.train(
    data="data.yaml",  # Path to the dataset configuration YAML file
    imgsz=640,            # Input image size (640x640)
    batch=32,             # Batch size for training
    epochs=100,           # Number of training epochs
    workers=0,            # Number of workers for data loading (reduce if encountering issues)
    device=0              # Device for training (0 for a single GPU; "cpu" for CPU training)
)


