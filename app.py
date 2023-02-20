import cv2
import time
import numpy as np
from helper import DamageHelper
from tracker import track

def is_overlap(bb1, bb2):
    xx = min(bb1[2], bb2[2]) > max(bb1[0], bb2[0])  # the smaller of the largest x-coordinates is larger than the larger of the smallest x-coordinates
    yy = min(bb1[3], bb2[3]) > max(bb1[1], bb2[1])  # the smaller of the largest y-coordinates is larger than the larger of the smallest y-coordinates
    return xx and yy

def tracking():
    model = DamageHelper('yolov5s6_openvino_model/yolov5s6.xml')
    roi = [530, 360, 1100, 365]
    cap = cv2.VideoCapture('video1.mp4')
    
    frames = []
    show_fps = False
    count = [[] for _ in range(8)]
    class_name = ['PERSON', 'BICYCLE', 'CAR', 'MOTORCYCLE', 'AIRPLANE', 'BUS', 'TRAIN', 'TRUCK']
    while cap.isOpened():
        ret, frame = cap.read()
        t = time.time()
        key = cv2.waitKey(1)
        if key == ord("q") or not ret:
            break
        elif key == ord('f'):
            show_fps = not show_fps

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(frame, 1280)
        if results is not None:
            results = [p for p in results if p[-1] < 8] # person, bicycle, car, motorcycle, airplane, bus, train, truck
        if results:
            results = np.array(results, dtype=float)
            detections = track(frame, results)
            for d in detections:
                x1, y1, x2, y2 = int(d.rect.x), int(d.rect.y), int(d.rect.max_x), int(d.rect.max_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f'{d.tracker_id}: {class_name[d.class_id]}', org=(x1+2, y1+15), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, 
                            color=(0, 255, 0), thickness=2)
                x, y = int(d.rect.center.x), int(d.rect.center.y)
                if is_overlap(roi, [x1, y1, x2, y2]):
                    # print('**** overlap ****')
                    if d.tracker_id not in count[d.class_id]:
                        count[d.class_id].append(d.tracker_id)
        
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 3)

        y=12
        for i,cls_name in enumerate(class_name):
            if count[i]:
                cv2.putText(frame, f'{cls_name}: {len(count[i])}', org = (0, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 255, 0), thickness=2)
                y += 15

        t = time.time() - t
        if show_fps and t != 0:
            cv2.putText(frame, f'FPS: {int(1/t)}', org=(0, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 255), thickness=2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        cv2.imshow('video', frame)
        
    cap.release()
    cv2.destroyAllWindows()

    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter('out1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

tracking()