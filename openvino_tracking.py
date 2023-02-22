import cv2
import time
import numpy as np
from tracker import track
from openvino.runtime import Core
from detector import Detector

class_name = ['PERSON', 'VEHICLE', 'BIKE']
roi = [530, 360, 1100, 365]
model_path = 'FP16/person-vehicle-bike-detection-crossroad-0078.xml'
video_path = 'video1.mp4'

def is_overlap(bb1, bb2):
    xx = min(bb1[2], bb2[2]) > max(bb1[0], bb2[0])  # the smaller of the largest x-coordinates is larger than the larger of the smallest x-coordinates
    yy = min(bb1[3], bb2[3]) > max(bb1[1], bb2[1])  # the smaller of the largest y-coordinates is larger than the larger of the smallest y-coordinates
    return xx and yy

def tracking():
    show_fps = False
    make_video = False
    frames = []
    count = [[] for _ in range(3)]
    detector = Detector(Core(), model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        t = time.time()
        key = cv2.waitKey(1)
        if key == ord("q") or not ret:
            break
        elif key == ord('f'):
            show_fps = not show_fps

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(frame)
        if results:
            results = np.array(results, dtype=float)
            detections = track(frame, results)
            for d in detections:
                if d.tracker_id is None:
                    continue
                x1, y1, x2, y2 = int(d.rect.x), int(d.rect.y), int(d.rect.max_x), int(d.rect.max_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{d.tracker_id}: {class_name[d.class_id]}', org=(x1+2, y1+15), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, 
                            color=(0, 255, 0), thickness=2)

                if is_overlap(roi, [x1, y1, x2, y2]):
                    if d.tracker_id not in count[d.class_id]:
                        count[d.class_id].append(d.tracker_id)

        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 3)

        y=28
        for i,cls_name in enumerate(class_name):
            if count[i]:
                cv2.putText(frame, f'{cls_name}: {len(count[i])}', org = (0, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 255, 0), thickness=2)
                y += 16

        t = time.time() - t
        if show_fps and t != 0:
            cv2.putText(frame, f'FPS: {int(1/t)}', org=(0, 12), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 255), thickness=2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        cv2.imshow('video', frame)

    cap.release()
    cv2.destroyAllWindows()

    if make_video:
        print('Making video ...')
        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter('out1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

tracking()