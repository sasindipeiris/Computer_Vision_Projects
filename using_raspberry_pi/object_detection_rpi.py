#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import datetime
from ultralytics import YOLO
import cv2
from imutils.video import VideoStream
import screeninfo

# define some constants
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)

model = YOLO("yolov5nu.pt")
#model = YOLO("./runs/detect/yolov8n_v8_50e2/weights/best.pt") # test trained model

print(model.names)

# initialize the video capture object
vs = VideoStream(src=0, resolution=(640, 640)).start()
#video_cap = cv2.VideoCapture(0)
#video_cap = cv2.VideoCapture("datasets\\Splash - 23011.mp4")
    
while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    frame = vs.read(); ret=True
    #ret, frame = video_cap.read()
    
    
    
    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]
    
    # loop over the detections
    #for data in detections.boxes.data.tolist():
    for box in detections.boxes:
        #extract the label name
        label=model.names.get(box.cls.item())
        
        # extract the confidence associated with the detection
        data=box.data.tolist()[0]
        confidence = data[4]

        # filter out weak detections
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

        #draw confidence and label
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(frame, "{} {:.1f}%".format(label,float(confidence*100)), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
       
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    #press key Q to exit
    if cv2.waitKey(1) == ord("q"):
        break

#video_cap.release()
vs.stop()
cv2.destroyAllWindows()