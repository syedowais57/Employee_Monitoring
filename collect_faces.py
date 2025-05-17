#!/usr/bin/env python3
import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture face images into a specified folder"
    )
    parser.add_argument(
        "--folder", required=True,
        help="Name of subfolder under face_db/custom to store images (e.g., f1, f2)"
    )
    parser.add_argument(
        "--source", default="rtsp://saibby:9906697711@192.168.1.14:554/stream1",
        help="Video source: '0' for webcam or RTSP URL"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Save one frame every N frames"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Determine video source (webcam index vs. URL) :contentReference[oaicite:4]{index=4}
    src = int(args.source) if args.source.isdigit() else args.source

    # Prepare output directory :contentReference[oaicite:5]{index=5}
    out_dir = os.path.join("face_db", "custom", args.folder)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(src)  # Open video stream :contentReference[oaicite:6]{index=6}
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    print(f"[INFO] Capturing to: {out_dir}. Press 'q' to quit.")
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No more frames. Exiting.")
            break

        cv2.imshow("Capture (press 'q' to quit)", frame)

        # Save every Nth frame :contentReference[oaicite:7]{index=7}
        if frame_count % args.interval == 0:
            img_path = os.path.join(out_dir, f"{saved_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1

        # Exit on 'q' :contentReference[oaicite:8]{index=8}
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {saved_count} images to {out_dir}")

if __name__ == "__main__":
    main()
