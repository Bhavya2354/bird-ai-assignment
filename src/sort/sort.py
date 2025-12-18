import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ---------------- IOU ---------------- #
def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    return inter / (area_test + area_gt - inter + 1e-6)


# ---------------- BBOX CONVERSIONS ---------------- #
def convert_bbox_to_z(bbox):
    w = max(bbox[2] - bbox[0], 1e-6)
    h = max(bbox[3] - bbox[1], 1e-6)
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / h
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    s = max(float(x[2]), 1e-6)
    r = max(float(x[3]), 1e-6)

    w = np.sqrt(s * r)
    h = s / w

    return np.array([
        x[0] - w / 2.,
        x[1] - h / 2.,
        x[0] + w / 2.,
        x[1] + h / 2.
    ]).reshape((1, 4))


# ---------------- TRACKER ---------------- #
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(convert_bbox_to_z(bbox))


# ---------------- SORT ---------------- #
class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        """
        dets: np.array [[x1,y1,x2,y2,conf], ...]
        returns: [[x1,y1,x2,y2,track_id], ...]
        """

        # Case 1: No detections
        if dets is None or len(dets) == 0:
            for trk in self.trackers:
                trk.predict()
            return np.empty((0, 5))

        # Case 2: No trackers yet
        if len(self.trackers) == 0:
            for i in range(len(dets)):
                self.trackers.append(KalmanBoxTracker(dets[i, :4]))
            return np.empty((0, 5))

        # Predict existing trackers
        trks = np.array([trk.predict()[0] for trk in self.trackers])

        # IoU computation + SAFETY
        iou_matrix = iou_batch(dets[:, :4], trks)
        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_trks = set()
        unmatched_dets = set(range(len(dets)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                self.trackers[c].update(dets[r, :4])
                matched_trks.add(c)
                unmatched_dets.discard(r)

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Collect valid tracks
        results = []
        alive_trackers = []

        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)

            if trk.hits >= self.min_hits:
                bbox = trk.predict()[0]
                results.append([*bbox, trk.id])

        self.trackers = alive_trackers
        return np.array(results)
