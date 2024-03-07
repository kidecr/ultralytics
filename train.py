from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('./ultralytics/cfg/models/v8/yolov8-fusion.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='./ultralytics/cfg/datasets/Anti-UAV.yaml', epochs=200, device=0, batch=48, cache=None, patience=30, data_mode="RGBT", save_period=100, resume=False)

# from ultralytics.models.yolo.detect import DetectionValidator

# args = dict(model='./ultralytics/cfg/models/v8/yolov8-fusion.yaml', data='./ultralytics/cfg/datasets/Anti-UAV.yaml')
# validator = DetectionValidator(args=args)
# validator()