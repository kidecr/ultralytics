from ultralytics import YOLO

############### val #################
# Load a model
# model = YOLO('./ultralytics/cfg/models/v8/yolov8-fusion.yaml')
model = YOLO('./runs/detect/train11/weights/best.pt')

# Customize validation settings
validation_results = model.val(data='./ultralytics/cfg/datasets/Anti-UAV.yaml', batch=16, conf=0.25,
                               iou=0.6, device=1, data_mode="RGBT3", ch=4)

# results = model.train(data='./ultralytics/cfg/datasets/Anti-UAV.yaml', epochs=1, device=1, batch=32, 
#                       cache=None, patience=100, data_mode="RGBT3", save_period=50, resume=False, 
#                       optimizer='AdamW', augment=True, lr0=1e-4, lrf=1e-5, freeze=[]) #[4,5,6,7,8,9,10,11,12, 14,15,16,17,18,19,20,21,22])
################# predict #####################

# # Load a pretrained YOLOv8n model
# import numpy as np
# model = YOLO('./runs/detect/train9/weights/best.pt')

# # Define path to the image file
# source = np.load('/home/wmh/Datasets/Anti-UAV-test.dataset/AntiUAV-RGBT-dataset/images/val/RGBT/2_62.npy') #'path/to/image.jpg'

# # Run inference on the source
# results = model(source)  # list of Results objects