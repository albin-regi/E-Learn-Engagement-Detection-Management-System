# E-Learn Engagement Detection & Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

A prototype Flask-based E-Learning Engagement Detection & Management System that uses a laptop camera to monitor student engagement during online classes. The project demonstrates real-time webcam capture, face/eye detection, a placeholder engagement scoring pipeline (replaceable with a trained ML model), and a web UI to view live video and engagement analytics.

**Note:** This README is written to be plug-and-play — copy the markdown and replace any placeholders (e.g. file names, model paths) with the actual values in your repository.

---

## Features

* Real-time webcam capture and display in browser
* Face and eye detection (Haar cascades / Mediapipe alternatives)
* Simple engagement scoring pipeline (dummy or model-based)
* Frontend pages (templates) for live feed and summary analytics
* Static assets (CSS/JS) for UI
* Configurable via `config.json`

---

## Quick Start (copy & paste)

```bash
# 1. Clone the repo (if not already):
# git clone https://github.com/albin-regi/E-Learn-Engagement-Detection-Management-System.git
cd E-Learn-Engagement-Detection-Management-System

# 2. Create virtual environment (recommended)
python3 -m venv venv
# Windows
# venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app (common variants; use the one your repo contains):
# If the repo uses Flask with app.py:
python app.py

# Or if the entry point is main.py:
python main.py

# Or if it is a Flask app named 'app' and you prefer to use flask run:
# export FLASK_APP=app.py   # or set FLASK_APP=app.py on Windows
# flask run --host=0.0.0.0 --port=5000
```

Open `http://localhost:5000` (or the port printed by the app) to view the interface.

---

## Prerequisites

* Python 3.8 or higher
* A laptop/desktop with a webcam
* (Optional) GPU + appropriate frameworks if you plan to swap in a deep learning model

Recommended Python packages (you will have these in `requirements.txt`, adjust as needed):

```text
flask
opencv-python
numpy
mediapipe  # optional, for landmark-based detection
tensorflow or torch  # optional, for ML models
scikit-learn
pillow
```

---

## Project Structure (example)

> The repository may contain variations of these filenames. Replace names below with the exact files in your repo when customizing.

```
E-Learn-Engagement-Detection-Management-System/
├─ app.py                # Flask entry point OR
├─ main.py               # alternative entry point for the application
├─ detector.py           # face/eye detection helpers (Haar cascades or Mediapipe wrappers)
├─ models.py             # engagement prediction functions / model loading
├─ utils.py              # utility helpers (draw overlays, helpers)
├─ config.json           # runtime configuration (camera index, thresholds)
├─ requirements.txt      # Python dependencies
├─ templates/
│  ├─ index.html         # live video page
│  └─ summary.html       # analytics / dashboard
├─ static/
│  ├─ css/
│  └─ js/
└─ README.md
```

---

## Configuration (`config.json`)

Include a `config.json` (or environment variables) to avoid hardcoding values. Example template you can copy:

```json
{
  "FLASK_HOST": "0.0.0.0",
  "FLASK_PORT": 5000,
  "DEBUG": true,
  "CAMERA_INDEX": 0,
  "FRAME_WIDTH": 640,
  "FRAME_HEIGHT": 480,
  "ENGAGEMENT_MODEL_PATH": "models/engagement_model.h5",
  "ENGAGEMENT_THRESHOLD": 0.5
}
```

Load it in Python with:

```python
import json
with open('config.json') as f:
    cfg = json.load(f)
CAMERA_INDEX = cfg.get('CAMERA_INDEX', 0)
```

---

## How it works (high-level)

1. **Web UI**: Flask serves a page (`/`) that embeds a video stream (server-sent frames via multipart JPEG or WebSocket).
2. **Video capture**: Server captures frames from the webcam using OpenCV (`cv2.VideoCapture(CAMERA_INDEX)`).
3. **Detection**: For each frame, a detection module finds faces and eyes using Haar cascades or Mediapipe face mesh.
4. **Engagement scoring**: A lightweight rule-based or ML model computes a per-frame engagement score (example: based on eye aspect ratio, head pose, gaze, attention landmarks).
5. **Overlay & stream**: The app overlays bounding boxes and the engagement score on the frame and streams it to the browser.
6. **Analytics**: The app logs per-frame or per-second engagement scores and exposes summary analytics on a dashboard page.

---

## Swap in a real ML model

The repository may include a dummy model (random/deterministic score). To use a trained model:

1. Train or obtain a model that accepts either raw frames or extracted features (e.g. eye aspect ratio, head pose, facial landmark vectors).
2. Save the model to `models/engagement_model.h5` (TensorFlow) or `models/engagement_model.pt` (PyTorch).
3. Modify `models.py` to load the model at startup and run inference inside the frame processing loop.

**Example (TensorFlow/Keras):**

```python
from tensorflow.keras.models import load_model
model = load_model('models/engagement_model.h5')

# prepare frame/features then:
score = model.predict(feature_vector[np.newaxis, :])[0][0]
```

**Important:** Keep inference fast — prefer lightweight models or pre-computed features.

---

## API / Endpoints (common)

* `GET /` — live video page
* `GET /video_feed` — multipart JPEG stream (used by `<img src="/video_feed">`)
* `GET /summary` — engagement metrics dashboard
* `POST /settings` — update runtime configuration (optional)

Adjust according to your actual app routes.

---

## Example: streaming frames (Flask + OpenCV)

```python
from flask import Response
import cv2

cap = cv2.VideoCapture(CAMERA_INDEX)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        # ... detect / overlay ...
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

---

## Troubleshooting

* **Black screen / camera not found**: Check `CAMERA_INDEX` (0, 1, ...). Ensure no other app is using the webcam.
* **Slow FPS / high CPU**: Reduce frame resolution, process every Nth frame, or move to a separate worker thread/process for inference.
* **Model too slow on CPU**: Use a smaller model (MobileNet variants), use ONNX runtime or TensorRT if GPU available.
* **Missing packages**: `pip install -r requirements.txt` or inspect `requirements.txt` for exact versions.

---

## Security & Privacy

* This project captures video from a webcam. Be mindful of privacy and legal requirements when using and sharing recordings.
* If deploying to public servers, never expose your webcam feed without authentication and encryption (HTTPS).

---

## Tests (Suggested)

Add a minimal test suite for utilities and detection code using `pytest`:

```
pip install pytest
pytest tests/
```

Create `tests/test_detector.py` to check that face/eye detectors return expected types when fed sample images.

---

## Deployment (local / server)

* For simple local use, run via `python app.py`.
* For production or multi-user, use a WSGI server like `gunicorn` (but note: capturing hardware cameras on cloud servers is not typical). Example:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

* If you need to support many remote students, consider client-side detection (run model in the browser via TensorFlow\.js) to avoid centralizing video streams.

---

## Contribution

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

Please include unit tests for new features.

---

## License

Include a license file in the repo. Example MIT license header to copy into `LICENSE`:

```
MIT License

Copyright (c) YEAR Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```

---

## Acknowledgements

* OpenCV (cv2)
* Haar cascades / Mediapipe for facial detection
* Inspiration: classroom engagement detection research

---

## Maintainer / Contact

If you want me to adapt this README to exactly match the repository file names and code (I can auto-detect entrypoints and edit the README), reply `Please customize` and I'll update it for you.

---

*End of README.*
