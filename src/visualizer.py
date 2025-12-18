import cv2
import os


class Visualizer:
    def __init__(self, output_path, fps, frame_size):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FPS safety
        if fps is None or fps <= 0:
            fps = 25

        self.frame_size = frame_size

        # Use MJPG codec (most reliable on Windows)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )

        if not self.writer.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open")

        self.output_path = output_path

    def draw(self, frame, tracks, count):
        # SAFETY: ensure frame matches writer size
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Bird Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        self.writer.write(frame)

    def release(self):
        self.writer.release()
