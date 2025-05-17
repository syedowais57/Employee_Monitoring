# utils/general_utils.py

import cv2
from datetime import datetime

def draw_label(frame, bbox, label, color=(0,255,0)):
    """Draw a labeled rectangle around a face/person."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y2-20), (x2, y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1+5, y2-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def log_activity(name, log_path="activity_log.txt"):
    """Append a timestamped recognition event to a log file."""
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(log_path, "a") as f:
        f.write(f"{ts} - {name}\n")
