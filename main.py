import cv2
import numpy as np

from yolo_segmentation import YOLOSegmentation
video = cv2.VideoCapture('video_2023-12-24_13-34-47.mp4')
ys = YOLOSegmentation(r"J:\AllProjectsPython\FaceTracker\runs\segment\train5\weights\best.pt")
while video.isOpened():
    ret, img = video.read()
    img = cv2.resize(img, (640, 640))
    bboxes, classes, segmentations, scores = ys.detect(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.medianBlur(img_gray, 31)
    all_res = list(zip(bboxes, classes, segmentations, scores))
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        mask = np.zeros_like(img_gray)
        cv2.fillPoly(mask, [seg], color=[255, 255, 255])
    gray_mask = cv2.bitwise_and(img_gray, mask)
    combined_image = np.where(mask == 255, img_gray, img)
    cv2.imshow("Image", combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()