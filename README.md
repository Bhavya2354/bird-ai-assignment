🐔 Bird Counting and Weight Estimation using Poultry CCTV

Author: Bhavya

Role Applied: Machine Learning / AI Engineer Intern

📌 Problem Statement

The objective of this assignment is to evaluate practical understanding of:

Machine Learning fundamentals

Computer Vision

Object detection and tracking

API development

The task is to build a prototype system that processes a fixed-camera poultry CCTV video and produces:

Bird Counts Over Time

Using object detection and stable tracking IDs

Bird Weight Estimation

Or a clearly defined weight proxy / index

Deliverables

Complete source code

Detailed README.md

Annotated output video

Sample JSON response from a FastAPI service

🧠 Solution Overview & Approach
Key Challenges

Provided dataset link was unavailable

Most public poultry datasets contain images, not videos

No ground-truth bird weight data available

Design Strategy

To address these constraints, the system was designed as an end-to-end CCTV analytics pipeline:

Images → Video → Detection → Tracking → Counting → Weight Proxy → Visualization → API

The focus is on correct system design and explainability, rather than only model training.

🛠 Technology Stack

Python 3.10+

YOLOv8 (Ultralytics) – object detection

OpenCV – video processing & annotation

SORT (Kalman Filter + IoU matching) – tracking

NumPy / SciPy – numerical computation

FastAPI + Uvicorn – API layer

✅ All tools are open-source
❌ No Docker used (as per instructions)
❌ No external paid APIs used
```
📁 Project Structure
bird-ai-assignemnt/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── app.py                    # FastAPI application
├── run_pipeline.py           # End-to-end pipeline runner
│
├── config/
│   └── config.yaml           # Thresholds, FPS, paths
│
├── models/
│   └── yolov8n.pt             # Pretrained YOLOv8 model
│
├── src/
│   ├── video_reader.py        # Image → video conversion
│   ├── detector.py            # YOLO inference wrapper
│   ├── tracker.py             # SORT-based tracking logic
│   ├── counter.py             # Bird counting logic
│   ├── weight_estimator.py    # Weight proxy estimation
│   ├── visualizer.py          # Annotated video writer
│   ├── utils.py               # Helper utilities
│   └── sort/
│       └── sort.py            # SORT tracker (MIT licensed)
│
├── data/
│   ├── images/
│   │   ├── train/images/
│   │   ├── valid/images/
│   │   └── test/images/
│   ├── sample_video.mp4       # Generated CCTV-style video
│   └── README.md              # Dataset source info
│
├── outputs/
│   ├── annotated_videos/
│   │   └── output.mp4         # Final annotated video
│   └── json/
│       └── sample_response.json
│
└── tests/
    ├── test_detector.py
    ├── test_tracking.py
    ├── test_counting.py
    └── test_weight.py
```
📊 Dataset Used

Since the dataset link provided in the task description was unavailable, the following public open-source dataset was used:

Roboflow – Chicken Detection Dataset

🔗 https://universe.roboflow.com/shashank-l4mfk/chicken-detection-ehuwm-jrr73

Dataset Details

Labeled poultry images

Train / validation / test split

Multiple poses and lighting conditions

Suitable for detection and tracking evaluation

🧩 Implementation Details

This section explains how each requirement was implemented.

1️⃣ Image → Video Conversion (CCTV Simulation)

File: src/video_reader.py

Dataset consists of static images

Images are:

Loaded in sorted order

Resized to a fixed resolution

Written sequentially into a video using OpenCV

Why this step is important

Enables realistic tracking behavior

Mimics fixed-camera poultry CCTV footage

Allows count-over-time analysis

Output

data/sample_video.mp4

2️⃣ Bird Detection

File: src/detector.py

Uses YOLOv8 pretrained model (yolov8n.pt)

Each frame produces:

Bounding boxes

Confidence scores

Only bird-related detections are retained

YOLOv8 was chosen for its speed, robustness, and generalization.

3️⃣ Bird Tracking (Stable IDs)

Files:

src/tracker.py

src/sort/sort.py

Tracking is implemented using SORT (Simple Online and Realtime Tracking), which combines:

Kalman Filter for motion prediction

IoU-based assignment for detection-to-track matching

Each bird receives a persistent ID, enabling identity preservation across frames.

4️⃣ Bird Counting Logic

File: src/counter.py

Counting is ID-based, not frame-based

Logic:

When a new tracking ID appears → count increases

Previously seen IDs are ignored

This prevents double-counting even if birds reappear.

5️⃣ Weight Estimation (Proxy / Index)

File: src/weight_estimator.py

Real bird weight ground truth is unavailable

A visual proxy is used:

Bounding box area ≈ relative bird size

Output is a weight index, not grams

This mirrors real-world poultry monitoring systems where visual estimation is used initially.

6️⃣ Visualization & Annotation

File: src/visualizer.py

Each frame is annotated with:

Bounding boxes

Tracking IDs

Current bird count

Weight proxy index

Annotated frames are written back into a video.

7️⃣ End-to-End Pipeline Execution

File: run_pipeline.py

This script:

Loads the generated video

Runs detection, tracking, counting, and weight estimation

Saves the annotated output video

Stores summary statistics for API usage

Command:

python run_pipeline.py

8️⃣ API Implementation (FastAPI)

File: app.py

Built using FastAPI

Exposes pipeline results via an HTTP endpoint

Endpoint

GET /analyze

📤 Output Explanation
🎥 Annotated Output Video

Path

outputs/annotated_videos/output.mp4


Contains

Bird bounding boxes

Unique tracking IDs

Bird count overlay

Weight proxy overlay

Purpose

Visual verification of detection & tracking

Easy inspection by reviewers

Demonstrates correctness of the system

📄 API JSON Output

Path

outputs/json/sample_response.json


Sample Response

{
  "total_birds_detected": 12,
  "frames_processed": 217,
  "average_weight_index": 0.74,
  "output_video": "outputs/annotated_videos/output.mp4"
}


Field Explanation

total_birds_detected → Unique birds counted using tracking IDs

frames_processed → Number of frames analyzed

average_weight_index → Mean relative weight proxy

output_video → Path to annotated video

📈 Accuracy & Validation Notes

Detection accuracy depends on YOLOv8 pretrained performance

Tracking stability validated via stable ID persistence

Counting correctness ensured through ID-based logic

Since labeled video ground truth is unavailable, qualitative validation and visual inspection were used, which is standard for prototype systems.

▶️ How to Run the Project (From Scratch)
git clone https://github.com/Bhavya2354/bird-ai-assignment.git
cd bird-ai-assignment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/video_reader.py
python run_pipeline.py
uvicorn app:app --reload

🧪 Testing
python tests/test_detector.py
python tests/test_tracking.py
python tests/test_counting.py
python tests/test_weight.py

📝 Notes

No Docker used

No external APIs used

Fully local and reproducible

Designed for clarity and explainability

✅ Conclusion

This project demonstrates:

Strong ML and computer vision fundamentals

Correct use of detection and tracking for analytics

Practical system design under real-world constraints

Clean, modular engineering

End-to-end ownership of a production-style prototype

Author: Bhavya

GitHub: https://github.com/Bhavya2354
