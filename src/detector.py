from ultralytics import YOLO
import cv2


class BirdDetector:
    def __init__(self, model_path: str, conf_thresh: float = 0.3):
        """
        YOLOv8-based bird detector
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns list of detections:
        [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 14 = bird
                if cls_id == 14:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, conf])

        return detections
