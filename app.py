from fastapi import FastAPI
import cv2

from src.detector import BirdDetector
from src.tracker import BirdTracker
from src.counter import BirdCounter
from src.weight_estimator import WeightEstimator

app = FastAPI(title="Bird Counting & Weight Estimation API")

VIDEO_PATH = "data/sample_video.mp4"
MODEL_PATH = "models/yolov8n.pt"


@app.get("/analyze")
def analyze_video():
    detector = BirdDetector(MODEL_PATH, conf_thresh=0.2)
    tracker = BirdTracker()
    counter = BirdCounter()
    weight_estimator = WeightEstimator(scale_factor=0.001)

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_idx = 0
    count_timeline = {}
    final_weights = {}
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        count = counter.update(frame_idx, tracks)
        weights = weight_estimator.estimate(tracks)

        count_timeline[frame_idx] = count

        for tid, w in weights.items():
            final_weights[tid] = w
            unique_ids.add(tid)

        frame_idx += 1

    cap.release()

    avg_weight = weight_estimator.average_weight(final_weights)

    return {
        "video": VIDEO_PATH,
        "total_birds_detected": len(unique_ids),
        "average_weight_index": avg_weight,
        "bird_weights": final_weights,
        "count_over_time": count_timeline
    }
