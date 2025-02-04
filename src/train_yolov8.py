from ultralytics import YOLO

model = YOLO('yolov8n.pt')  #скачать веса на сайте ultralytics

model.train(data='path/to/yolo_data.yaml', epochs=100)

model.export(format='pt',save=True,name='yolo8_detector')
