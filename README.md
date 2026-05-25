# Pedestrian Detection Using HOG Feature Extraction

## Overview
This project detects pedestrians in images using HOG (Histogram of Oriented Gradients) feature extraction and OpenCV.

The system uses:
- HOG Descriptor
- SVM Detector
- Non-Maximum Suppression (NMS)

to identify pedestrians accurately in different lighting conditions and environments.

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Imutils

---

## Features
- Pedestrian detection in images
- Bounding box visualization
- Confidence threshold filtering
- Overlapping box removal using NMS
- Low-light image testing support

---

## Project Structure

```text
Pedestrian_Detection_Using_HOG_Feature_Extraction/
│
├── pedestrian_test.py
├── no pedes.jpeg
├── pedes no 1.jpeg
├── pedes.jpg
├── low light.jpeg
├── requirements.txt
├── README.md
```

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python pedestrian_test.py
```

---

## Working Process
1. Image is loaded using OpenCV
2. HOG features are extracted
3. SVM detector identifies pedestrians
4. Non-Maximum Suppression removes overlapping boxes
5. Final detection is displayed

---

## Output
Detected pedestrians are highlighted using green bounding boxes.

---

## Author
Thrivikram Chakradhar
