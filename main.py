import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('highway2.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

counted_cars = 0

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    roi = frame[322:713, 194:614]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.line(frame, (150, 495), (575, 495), (0, 255, 255), 2)
        if area > 3000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            print(cv2.contourArea(cnt))
            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if (1 < x < 382) & (y >= 175) & (cv2.contourArea(cnt) >= 25):
                if (y >= 185) & (cv2.contourArea(cnt)<40):
                    break
                counted_cars += 1
            print(detections)
    cv2.putText(frame, "Naliczone samochody: " + str(counted_cars), (589,366), cv2.FONT_HERSHEY_PLAIN, 1, (0,180,0),2)

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('roi', roi)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()