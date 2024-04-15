from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('./ultralytics/cfg/models/v8/yolov8-fusion.yaml')
# model = YOLO('./rgbt-update.pt')
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='./ultralytics/cfg/datasets/Anti-UAV.yaml', epochs=200, device=2, batch=32, 
                      cache=None, patience=100, data_mode="RGBT3", save_period=50, resume=False, 
                      optimizer='AdamW', augment=True, lr0=1e-4, lrf=1e-5, freeze=[]) #[4,5,6,7,8,9,10,11,12, 14,15,16,17,18,19,20,21,22])

# from ultralytics.models.yolo.detect import DetectionValidator

# args = dict(model='./ultralytics/cfg/models/v8/yolov8-fusion.yaml', data='./ultralytics/cfg/datasets/Anti-UAV.yaml')
# validator = DetectionValidator(args=args)
# validator()
