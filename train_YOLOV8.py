from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 nano (léger), sinon essaye yolov8s.pt

model.train(data="TACO-YOLO/dataset.yaml", epochs=2, batch=16, imgsz=640)
