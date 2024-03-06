from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('./ultralytics/cfg/models/v8/yolov8-fusion.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='./ultralytics/cfg/datasets/Anti-UAV.yaml', epochs=200, device=2, batch=16, cache="ram", patience=20, data_mode="RGB2", save_period=10, resume=False)

# from ultralytics.models.yolo.detect import DetectionValidator

# args = dict(model='./ultralytics/cfg/models/v8/yolov8-fusion.yaml', data='./ultralytics/cfg/datasets/Anti-UAV.yaml')
# validator = DetectionValidator(args=args)
# validator()