# BlindAssist

## Real-Time Object Detection, Face Recognition and Emotion-Based Voice Assistant

BlindAssist is a real-time assistive system designed to help visually impaired users understand their surroundings through a webcam. The system combines object detection, face recognition, emotion recognition, distance estimation, direction detection, and offline voice output.

## Features

* Real-time object detection using YOLO
* Detection of everyday objects such as people, bottles, cups, books, phones, etc.
* Custom object detection support for objects such as pens and pencils
* Face detection and recognition for known people
* Emotion recognition for detected/recognized faces
* Detects emotions such as happy, sad and neutral
* Voice output using offline text-to-speech
* Direction indication: left, ahead or right
* Approximate distance estimation for selected objects
* Repeated speech prevention for the same continuously visible object
* Object disappearance/reappearance handling
* Webcam-based operation using OpenCV
* CUDA support when a compatible GPU is available

## System Workflow

Webcam
↓
OpenCV Frame Capture
↓
YOLO Object Detection
↓
Object Filtering & Tracking
↓
Direction and Distance Estimation
↓
Face Detection
↓
Face Recognition
↓
Emotion Recognition
↓
Speech Queue
↓
Offline Voice Output

## Technologies Used

* Python
* OpenCV
* YOLOv8
* Ultralytics
* YuNet Face Detection
* SFace Face Recognition
* HSEmotion
* ONNX Runtime
* pyttsx3
* NumPy

## Project Structure

```text
BlindAssist/
│
├── main.py
├── requirements.txt
├── README.md
│
├── known_faces/
│   ├── person1.jpg
│   └── person2.jpg
│
├── face_models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
└── yolov8n.pt
```

> File names may differ depending on the current implementation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CoderNishchay/BlindAssist1.git
cd BlindAssist1
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the main Python file:

```bash
python main.py
```

Make sure your webcam is connected and accessible by OpenCV.

## Known Faces

Place photographs of known people inside the `known_faces` folder.

For example:

```text
known_faces/
├── nishchay.jpg
├── chanchal.jpg
└── sapna.jpg
```

The system uses these images to recognize known people.

## Object Detection

BlindAssist uses YOLO for real-time object detection.

The system can work with the standard YOLO model and can also be extended with a custom-trained model for objects that are not reliably detected by the default model.

Examples of useful custom classes include:

* Pen
* Pencil
* Glasses
* Earphones
* Charger
* Notebook

## Direction Detection

The camera frame is divided into three regions:

```text
+----------------+----------------+----------------+
|      LEFT      |     AHEAD      |      RIGHT     |
+----------------+----------------+----------------+
```

The position of the detected object's bounding box determines whether the system reports the object as being on the left, ahead, or right.

## Distance Estimation

The system provides approximate distance estimation for selected objects using bounding-box dimensions and predefined focal-length values.

Distance estimation is intended as an approximate indication rather than an exact measurement.

## Voice Assistance

BlindAssist uses `pyttsx3` for offline text-to-speech.

This allows the system to provide spoken information without requiring an internet connection.

The speech system uses a separate queue/thread so that voice output does not unnecessarily block the real-time detection loop.

## Emotion Recognition

For recognized faces, the system can estimate facial emotion.

The current implementation can report emotions such as:

* Happy
* Sad
* Neutral

Example:

```text
Nishchay is happy.
```

## Safety-Oriented Object Categories

The system can be extended to identify potentially important or dangerous objects, including:

* Knife
* Scissors
* Blade
* Cutter
* Needle
* Syringe
* Broken glass
* Hammer

These detections are intended to provide an additional awareness layer for the user.

## Requirements

* Windows/Linux computer
* Python 3.11 recommended
* Webcam
* Sufficient CPU/GPU resources for real-time inference
* Compatible OpenCV installation

A CUDA-compatible GPU can be used when available to improve inference performance.

## Limitations

* Distance estimation is approximate.
* Object detection performance depends on lighting, camera quali
