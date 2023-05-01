from ultralytics import YOLO
import numpy as np
import cv2
import pafy
import time

url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
video = pafy.new(url)
best = video.getbest()

cap = cv2.VideoCapture(best.url)
model = YOLO("yolov3-tiny.pt")
starting_time = time.time()
frame_id = 0
colors = np.random.uniform(0, 255, size=(255, 3))

while True:
    ret, frame = cap.read()
    frame_id+=1
    if not ret:
        break
    results = model(frame, device="mps")
    result = results[0] 
       
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        
        cv2.rectangle(frame, (x, y), (x2, y2), colors[cls], 2)
        cv2.putText(frame, str(model.names[int(cls)]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, colors[cls], 2)
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 255), 3)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
