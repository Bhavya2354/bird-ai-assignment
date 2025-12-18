🐔 Bird Counting & Weight Estimation using Computer Vision
📌 Project Overview

This project is a computer vision–based prototype developed to analyze poultry farm CCTV footage using object detection, tracking, and analytics.

The system processes a fixed-camera poultry video and provides:

🐓 Bird counting over time using stable tracking IDs

⚖️ Bird weight estimation using a valid proxy metric

🎥 Annotated output video for visual verification

🌐 FastAPI service that returns structured analytics in JSON format

The focus of this assignment is correctness, explainability, and clean engineering, rather than cloud deployment or large-scale infrastructure.

🎯 Problem Statement (Assignment Context)

Given a poultry farm CCTV feed, build a prototype that can:

Detect birds in each frame

Track birds across frames with stable IDs

Count birds over time (not per frame only)

Estimate bird weight (or a defined proxy)

Expose the results through an API

This project fulfills all requirements mentioned in the assignment PDF.

🧠 Design Approach & Key Decisions
🔹 Bird Detection

YOLOv8 (Ultralytics) is used for bird detection

A lightweight pretrained model (yolov8n.pt) is used for fast inference

Detection runs on CPU, no GPU dependency

🔹 Bird Tracking

SORT (Simple Online and Realtime Tracking) algorithm is implemented

Uses:

Kalman Filter for motion prediction

IoU-based matching for detection–track association

Each bird is assigned a stable tracking ID

🔹 Bird Counting Logic

A bird is counted once per unique tracking ID

Count is tracked over time (frame-wise)

This avoids double counting and ensures temporal consistency

🔹 Weight Estimation (Proxy-Based)

⚠️ Important Note:
True bird weight cannot be measured using a single RGB CCTV camera.

So a weight proxy is used:

Bounding box area is treated as a relative indicator of bird size

Larger bounding box → higher weight index

This approach is commonly used in poultry monitoring systems

The output is a relative weight index, not grams or kilograms.

🔹 API Design

Implemented using FastAPI

Designed for local evaluation

Returns structured JSON analytics

A sample API response is included for easy review

📁 Project Structure
bird-ai-assignment/
├── app.py                    # FastAPI application
├── run_pipeline.py           # End-to-end video processing script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── src/                      # Core logic
│   ├── detector.py           # YOLOv8 bird detection
│   ├── tracker.py            # SORT-based tracking
│   ├── counter.py            # Bird counting over time
│   ├── weight_estimator.py   # Weight proxy estimation
│   ├── visualizer.py         # Annotated video generation
│   └── utils.py              # Helper utilities
│
├── models/
│   └── yolov8n.pt             # Pretrained YOLOv8 model
│
├── data/
│   └── sample_video.mp4       # Input video (generated from images)
│
├── outputs/
│   ├── annotated_videos/
│   │   └── output.avi         # Annotated output video
│   └── json/
│       └── sample_response.json  # Sample API JSON response

📦 Dataset Details
🔹 Dataset Source

Roboflow Universe – Chicken Detection Dataset

Dataset contains images only, no videos

🔹 Image-to-Video Conversion

Since no video dataset was available:

Sequential images were stitched into a video

This simulates a fixed CCTV camera feed

Resulting file:

data/sample_video.mp4


This approach is common when working with surveillance-style datasets.

🛠️ Installation & Setup
✅ Prerequisites

Python 3.9 or above

Works on Windows / Linux / macOS

No GPU required

🔹 Step 1: Clone Repository
git clone <your-github-repo-link>
cd bird-ai-assignment

🔹 Step 2: Create Virtual Environment
python -m venv venv


Activate:

Windows

.\venv\Scripts\activate


Linux / macOS

source venv/bin/activate

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

▶️ How to Run the Project
🟢 1. Generate Annotated Output Video
python run_pipeline.py


This generates:

outputs/annotated_videos/output.avi


The video includes:

Bird bounding boxes

Tracking IDs

Live bird count overlay

🟢 2. Run the FastAPI Service
uvicorn app:app --host 127.0.0.1 --port 8000

🟢 3. Access the API
Swagger UI (Recommended)
http://127.0.0.1:8000/docs

Direct API Endpoint
http://127.0.0.1:8000/analyze

📤 API Output

A sample API response is saved at:

outputs/json/sample_response.json


The response contains:

Total birds detected

Count over time (frame-wise)

Per-bird weight index

Average weight index

This allows reviewers to inspect results without running the code.

📝 Notes for Evaluators

The API is intended for local evaluation

Annotated video and sample JSON are included for easy validation

Weight estimation is a proxy, not a physical measurement

The project is modular and easy to extend

🚀 Possible Improvements

Camera calibration for real-world scaling

Depth estimation using monocular or stereo vision

Multi-camera tracking

Temporal smoothing for weight trends

Live RTSP stream integration

✅ Conclusion

This prototype demonstrates:

Practical application of computer vision techniques

Correct use of detection, tracking, and temporal analytics

Honest handling of real-world constraints

Clean and reproducible engineering practices

👤 Author

Bhavya

Machine Learning / Computer Vision Intern Applicant
