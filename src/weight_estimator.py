import numpy as np


class WeightEstimator:
    """
    Estimates bird weight using bounding box area as a proxy.
    Returns a relative weight index, not absolute grams.
    """

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def estimate(self, tracks):
        """
        tracks: [[x1, y1, x2, y2, track_id], ...]

        returns:
        {
            track_id: weight_index
        }
        """

        weights = {}

        for t in tracks:
            x1, y1, x2, y2, track_id = t
            area = max((x2 - x1) * (y2 - y1), 0)

            weight_index = area * self.scale_factor
            weights[track_id] = round(float(weight_index), 2)

        return weights

    def average_weight(self, weights):
        """
        weights: dict of {track_id: weight_index}
        """
        if not weights:
            return 0.0
        return round(float(np.mean(list(weights.values()))), 2)
