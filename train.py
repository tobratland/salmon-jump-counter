from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


save_dir = '/home/tobratland/Workspace/salmon-detector/runs'
# Use the model
model.train(data="dataset.yaml", epochs=300, batch=65, save_dir=save_dir) 