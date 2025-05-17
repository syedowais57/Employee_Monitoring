#!/usr/bin/env python3
"""
Day 1: Capture and display frames from a camera or video file.
Usage:
  # IP camera (RTSP):
  python day1_capture.py --source rtsp://<user>:<pass>@<ip>/stream

  # Local webcam:
  python day1_capture.py --source 0

  # Video file:
  python day1_capture.py --source /path/to/video.mp4
"""

import cv2
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Day 1: Camera/Video Capture")
    parser.add_argument(
        "--source", required=True,
        help="rtsp://saibbyweb:9906697711Sw@192.168.1.14:554/stream1"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src = args.source

    # interpret numeric source as webcam index
    try:
        src = int(src)
    except ValueError:
        pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Successfully opened source: {args.source}")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received. Exiting.")
            break

        # Display the frame
        cv2.imshow("Day 1 Capture", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
