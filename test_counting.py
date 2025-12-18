import cv2
from src.detector import BirdDetector
from src.tracker import BirdTracker
from src.counter import BirdCounter

detector = BirdDetector("models/yolov8n.pt", conf_thresh=0.2)
tracker = BirdTracker()
counter = BirdCounter()

cap = cv2.VideoCapture("data/sample_video.mp4")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    count = counter.update(frame_idx, tracks)

    print(f"Frame {frame_idx} | Count: {count}")
    frame_idx += 1
