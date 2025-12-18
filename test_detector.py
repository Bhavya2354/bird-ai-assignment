import cv2
from src.detector import BirdDetector

detector = BirdDetector("models/yolov8n.pt")

cap = cv2.VideoCapture("data/sample_video.mp4")
ret, frame = cap.read()

if not ret:
    print("Failed to read frame")
else:
    detections = detector.detect(frame)
    print("Detections:", detections)
