

import cv2
from pathlib import Path

# Find the models folder relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

yunet_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
sface_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# Check that both files exist
print("YuNet file exists:", yunet_path.exists())
print("SFace file exists:", sface_path.exists())

# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    str(yunet_path),
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# Load SFace face recognizer
recognizer = cv2.FaceRecognizerSF.create(
    str(sface_path),
    ""
)

print("YuNet loaded successfully!")
print("SFace loaded successfully!")
