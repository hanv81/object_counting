import cv2
import time
import numpy as np
from tracker import track
from openvino.runtime import Core
from detector import Detector
from shapely.geometry import LineString,Polygon

class_name = ['PERSON', 'VEHICLE', 'BIKE']
roi_x1, roi_y1, roi_x2, roi_y2 = 530, 400, 1030, 355
roi_line = LineString([(roi_x1, roi_y1), (roi_x2, roi_y2)])
model_path = 'FP16/person-vehicle-bike-detection-crossroad-0078.xml'
video_path = 'video1.mp4'
make_video = False
draw_model_detect = False

def is_overlap(line, bbox):
    x1,y1,x2,y2 = bbox
    return line.intersects(Polygon(((x1,y1), (x2,y1), (x2,y2), (x1,y2))))

def tracking():
    show_fps = False
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
            if draw_model_detect:
                for x1, y1, x2, y2, _, _ in results:
                    cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 0), 2)
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

                if is_overlap(roi_line, [x1, y1, x2, y2]):
                    if d.tracker_id not in count[d.class_id]:
                        count[d.class_id].append(d.tracker_id)

        cv2.line(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,0), 3)

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
    for i,cls_name in enumerate(class_name):
        print(f'{cls_name}: {len(count[i])}')

    if make_video:
        print('Making video ...')
        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter('out1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

tracking()