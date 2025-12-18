import cv2
from src.detector import BirdDetector
from src.tracker import BirdTracker
from src.counter import BirdCounter
from src.weight_estimator import WeightEstimator
from src.visualizer import Visualizer

VIDEO_PATH = "data/sample_video.mp4"
OUTPUT_PATH = "outputs/annotated_videos/output.avi"

detector = BirdDetector("models/yolov8n.pt", conf_thresh=0.2)
tracker = BirdTracker()
counter = BirdCounter()
weight_estimator = WeightEstimator(scale_factor=0.001)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    raise RuntimeError("Invalid video frame size")


viz = Visualizer(OUTPUT_PATH, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    count = counter.update(frame_idx, tracks)
    weights = weight_estimator.estimate(tracks)

    viz.draw(frame, tracks, count)
    frame_idx += 1

cap.release()
viz.release()
print("Annotated video saved:", OUTPUT_PATH)
