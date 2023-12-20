import ultralytics
import cv2
import numpy as np
import cvzone

model = ultralytics.YOLO('./runs/detect/train3/weights/best.pt')

# Define current screenshot as source
source = '1.jpg'
frame = cv2.imread(source)
# Run inference on the source
results = model(frame, classes=[1])
for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=5)

            # currentArray = np.array((x1, y1, x2, y2))
            # detections = np.vstack((detections, currentArray))
cv2.imshow('Look', frame)
cv2.waitKey(0)
