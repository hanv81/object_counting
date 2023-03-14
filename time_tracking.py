import cv2
import time
import numpy as np
from helper import DamageHelper
from tracker import track
from shapely.geometry import Polygon, Point

roi = np.array([[250, 200], [440, 200], [320, 400], [70, 400]])
roi_pts = roi.reshape((-1,1,2))
roi_polygon = Polygon(roi)
model = DamageHelper('yolov5s6_openvino_model/yolov5s6.xml')
frame_size = 1280
video_path = 'supermarket.mp4'
make_video = False
draw_model_detect = False

def is_overlap(pt):
    return roi_polygon.contains(Point(pt))

def tracking():
    frames = []
    show_fps = False
    cap = cv2.VideoCapture(video_path)
    track_time = {}
    while cap.isOpened():
        ret, frame = cap.read()
        t = time.time()
        key = cv2.waitKey(1)
        if key == 27 or not ret:
            break
        elif key == ord('f'):
            show_fps = not show_fps

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(frame, frame_size)
        if results is not None:
            results = [p for p in results if p[-1] == 0] # person only
        if results:
            if draw_model_detect:
                for x1, y1, x2, y2, _, _ in results:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            results = np.array(results, dtype=float)
            detections = track(frame, results)
            for d in detections:
                x1, y1 = int(d.rect.x), int(d.rect.y)
                x, y = int(d.rect.center.x), int(d.rect.center.y)
                obj_time = track_time.get(d.tracker_id)
                if is_overlap((x,y)):
                    if obj_time is None:
                        track_time[d.tracker_id] = [time.time(), 0]

                    start, total = track_time[d.tracker_id]
                    if start is None:
                        start = time.time()
                        track_time[d.tracker_id][0] = start
                    obj_time = round(total + time.time() - start, 2)

                elif obj_time is not None:
                    start, total = obj_time
                    if start is not None:
                        total += (time.time() - start)
                        track_time[d.tracker_id] = [None, total]
                    obj_time = round(total, 2)
                
                cv2.circle(frame, (x, y), radius=1, color=(255, 0, 0), thickness=2)
                cv2.putText(frame, f'{d.tracker_id}: {obj_time}', org=(x1+2, y1+15), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, 
                            color=(0, 255, 0), thickness=2)

        cv2.polylines(frame, [roi_pts], True, (0,255,0), 3)

        t = time.time() - t
        if show_fps and t != 0:
            cv2.putText(frame, f'FPS: {int(1/t)}', org=(0, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 255), thickness=2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        cv2.imshow('video', frame)
        
    cap.release()
    cv2.destroyAllWindows()

    if make_video:
        print('Making video ...')
        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

tracking()