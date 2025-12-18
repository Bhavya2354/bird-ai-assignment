from collections import defaultdict


class BirdCounter:
    """
    Counts unique birds over time using tracking IDs
    """

    def __init__(self):
        self.seen_ids = set()
        self.time_series = defaultdict(int)

    def update(self, frame_idx, tracks):
        """
        frame_idx: int (frame number or timestamp)
        tracks: [[x1, y1, x2, y2, track_id], ...]

        Returns current total unique bird count
        """

        for track in tracks:
            track_id = track[4]
            self.seen_ids.add(track_id)

        self.time_series[frame_idx] = len(self.seen_ids)
        return self.time_series[frame_idx]

    def get_counts(self):
        """
        Returns count over time
        """
        return dict(self.time_series)
