from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics.yolo.utils.plotting import Annotator
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
results = model.train(data='coco128.yaml', epochs=3)
results = model.val()
cap = cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,920)

fig, ax = plt.subplots(figsize=(10,6))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results= model(frame ,stream=True)
    for r in results:
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names)
          
    frame = annotator.result()  
    ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.pause(0.1)
    ax.clear()

    if cv2.waitKey(1) == ord("1"):
        break
cap.release()
plt.close()
