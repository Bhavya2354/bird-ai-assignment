import cv2
from src.detector import BirdDetector
from src.tracker import BirdTracker

detector = BirdDetector("models/yolov8n.pt", conf_thresh=0.2)
tracker = BirdTracker()

cap = cv2.VideoCapture("data/sample_video.mp4")

for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    print(tracks)
