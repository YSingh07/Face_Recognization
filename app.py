
from flask import Flask, render_template, request, jsonify
from database import register_person, get_all_people




import cv2
import numpy as np
from pathlib import Path
from flask import request, jsonify

# Model file paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# Load the models once when Flask starts
detector = cv2.FaceDetectorYN.create(
    str(YUNET_PATH),
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_PATH),
    ""
)

print("Face detection and recognition models are ready!")


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    username = data.get("username", "").strip()
    embeddings = data.get("embeddings")

    if not username:
        return jsonify({
            "success": False,
            "message": "Username cannot be empty."
        }), 400

    if not isinstance(embeddings, list) or len(embeddings) < 5:
        return jsonify({
            "success": False,
            "message": "At least 5 face samples are required."
        }), 400

    # Every sample must be a numerical embedding
    for sample in embeddings:
        if (
            not isinstance(sample, list)
            or not sample
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in sample
            )
        ):
            return jsonify({
                "success": False,
                "message": "Invalid face embedding data."
            }), 400

    try:
        person_id = register_person(username, embeddings)

        if person_id is None:
            return jsonify({
                "success": False,
                "message": "That username is already registered."
            }), 409

        return jsonify({
            "success": True,
            "message": f"{username} registered successfully!",
            "id": person_id
        })

    except Exception:
        app.logger.exception("Registration failed")
        return jsonify({
            "success": False,
            "message": "Database error while registering the person."
        }), 500


@app.route("/api/people", methods=["GET"])
def people():
    try:
        rows = get_all_people()

        return jsonify({
            "success": True,
            "people": [
                {
                    "username": username,
                    "samples": len(embeddings)
                }
                for username, embeddings in rows
            ]
        })

    except Exception:
        app.logger.exception("Could not load registered people")
        return jsonify({
            "success": False,
            "message": "Could not load registered people."
        }), 500



@app.route("/api/face_embedding", methods=["POST"])
def face_embedding():
    uploaded_file = request.files.get("frame")

    if uploaded_file is None:
        return jsonify(success=False, message="No webcam frame received."), 400

    # Convert the uploaded image into an OpenCV image
    image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify(success=False, message="Invalid image."), 400

    # YuNet needs the actual image dimensions
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return jsonify(success=False, message="No face detected."), 422

    if len(faces) > 1:
        return jsonify(
            success=False,
            message="Multiple faces detected. Show only one face."
        ), 422

    # Align and crop the detected face
    aligned_face = recognizer.alignCrop(frame, faces[0])

    # Generate a 128-dimensional face embedding
    embedding = recognizer.feature(aligned_face).flatten().tolist()

    return jsonify(
        success=True,
        message="Face detected successfully!",
        embedding=embedding
    )



@app.route("/api/recognize", methods=["POST"])
def recognize():
    uploaded_file = request.files.get("frame")

    if uploaded_file is None:
        return jsonify(
            success=False,
            message="No webcam frame received."
        ), 400

    # Convert the uploaded image into an OpenCV image
    image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify(success=False, message="Invalid image."), 400

    # Detect the face
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return jsonify(
            success=False,
            recognized=False,
            message="No face detected."
        )

    if len(faces) > 1:
        return jsonify(
            success=False,
            recognized=False,
            message="Multiple faces detected. Show one face."
        )

    # Generate the live face embedding
    aligned_face = recognizer.alignCrop(frame, faces[0])
    live_embedding = recognizer.feature(aligned_face).flatten()

    # Retrieve registered people and their saved embeddings
    people = get_all_people()

    if not people:
        return jsonify(
            success=True,
            recognized=False,
            message="No registered people found."
        )

    threshold = 0.363
    best_username = None
    best_similarity = -1.0

    # Compare against every saved sample
    for username, saved_embeddings in people:
        similarities = []

        for saved_embedding in saved_embeddings:
            saved_vector = np.asarray(
                saved_embedding, dtype=np.float32
            ).flatten()

            if saved_vector.shape != live_embedding.shape:
                continue

            denominator = (
                np.linalg.norm(live_embedding)
                * np.linalg.norm(saved_vector)
            )

            if denominator == 0:
                continue

            similarity = float(
                np.dot(live_embedding, saved_vector) / denominator
            )

            similarities.append(similarity)

        if not similarities:
            continue

        # Average the three highest similarities for this person
        similarities.sort(reverse=True)
        top_scores = similarities[:3]
        person_score = sum(top_scores) / len(top_scores)

        if person_score > best_similarity:
            best_similarity = person_score
            best_username = username

    if best_username is not None and best_similarity >= threshold:
        return jsonify(
            success=True,
            recognized=True,
            username=best_username,
            similarity=round(best_similarity, 4),
            message=f"Recognized: {best_username}"
        )

    return jsonify(
        success=True,
        recognized=False,
        similarity=round(best_similarity, 4),
        message="Unknown person."
    )



if __name__ == "__main__":
    app.run(debug=True)
