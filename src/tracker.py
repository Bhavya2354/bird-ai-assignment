import numpy as np
from src.sort.sort import Sort


class BirdTracker:
    """
    Wrapper around SORT tracker to track birds with stable IDs
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

    def update(self, detections):
        """
        detections: list of detections
        Format: [[x1, y1, x2, y2, confidence], ...]

        returns: tracked objects
        Format: [[x1, y1, x2, y2, track_id], ...]
        """

        # If no detections, just update tracker
        if detections is None or len(detections) == 0:
            tracks = self.tracker.update(np.empty((0, 5)))
            return tracks.tolist() if len(tracks) > 0 else []

        # Convert detections to numpy array
        dets = np.array([
            [d[0], d[1], d[2], d[3], d[4]]
            for d in detections
        ])

        tracks = self.tracker.update(dets)

        results = []
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            results.append([float(x1), float(y1), float(x2), float(y2), int(track_id)])

        return results
