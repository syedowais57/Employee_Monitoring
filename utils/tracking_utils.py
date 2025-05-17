import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 detector once
detector = YOLO("yolov8n.pt")
# Initialize DeepSORT tracker once
tracker  = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

def track_frame(frame):
    """
    Detects people in `frame` with YOLOv8, tracks them with DeepSORT,
    and returns list of (track_id, x1, y1, x2, y2).
    """
    # 1) Detect persons only
    results = detector(frame, classes=[0])[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # 2) Update tracks
    tracks = tracker.update_tracks(detections, frame=frame)

    # 3) Build output list
    outputs = []
    for t in tracks:
        l, t_, r, b = t.to_ltrb()
        outputs.append((t.track_id, int(l), int(t_), int(r), int(b)))
    return outputs
