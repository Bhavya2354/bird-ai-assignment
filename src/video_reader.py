import cv2
import os
from pathlib import Path


def images_to_video(
    image_dir: str,
    output_path: str,
    fps: int = 5,
    max_frames: int = 300
):
    """
    Convert a folder of images into a fixed-camera CCTV-style video.
    """

    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted([
        img for img in image_dir.iterdir()
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(images) == 0:
        raise ValueError(f"No images found in directory: {image_dir}")

    images = images[:max_frames]  # limit video length

    print(f"Found {len(images)} images")
    print(f"Creating video at: {output_path}")

    first_frame = cv2.imread(str(images[0]))
    height, width, _ = first_frame.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    for img_path in images:
        frame = cv2.imread(str(img_path))
        frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print("Video generation completed successfully!")


if __name__ == "__main__":
    images_to_video(
        image_dir="data/images/train/images",
        output_path="data/sample_video.mp4",
        fps=5,
        max_frames=300
    )
