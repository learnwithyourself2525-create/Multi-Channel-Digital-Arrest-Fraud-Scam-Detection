# pipeline/detection_pipeline.py
import subprocess
from models.text_classifier import TextClassifier
from models.audio_processor import AudioProcessor
from models.video_deepfake_detector import VideoDeepfakeDetector
import numpy as np
import cv2

# Initialize models with error handling to prevent silent crashes
try:
    print("Initializing TextClassifier...")
    text_classifier = TextClassifier()
    print("TextClassifier initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize TextClassifier: {e}")
    text_classifier = None

try:
    print("Initializing AudioProcessor (Whisper)... this may take a moment.")
    audio_processor = AudioProcessor()
    print("AudioProcessor initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize AudioProcessor: {e}")
    audio_processor = None

try:
    print("Initializing VideoDeepfakeDetector (DeepFace)... this may take a moment.")
    video_detector = VideoDeepfakeDetector()
    print("VideoDeepfakeDetector initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize VideoDeepfakeDetector: {e}")
    video_detector = None


async def process_text_input(text: str) -> dict:
    if not text_classifier:
        return {"type": "text_analysis", "result": {"error": "Text model is not available."}}
    prediction = text_classifier.predict(text)
    return {"type": "text_analysis", "result": prediction}

# pipeline/detection_pipeline.py

async def process_audio_input(audio_path: str) -> dict:
    if not audio_processor or not text_classifier:
        return {"type": "audio_analysis", "result": {"error": "Audio/Text model not available."}}

    audio_result = audio_processor.process_audio(audio_path)
    if "error" in audio_result:
        return {"type": "audio_analysis", "result": audio_result}

    transcribed_text = audio_result.get("transcribed_text", "")
    text_prediction = text_classifier.predict(transcribed_text)

    return {
        "type": "audio_analysis",
        "result": {
            "transcription": transcribed_text,
            "scam_analysis": text_prediction
        }
    }


async def process_video_file(video_path: str) -> dict:
    if not video_detector:
        return {"type": "video_analysis", "result": {"error": "Video model not available."}}

    # Extract audio track
    audio_path = f"{video_path}_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            check=True
        )
    except Exception as e:
        audio_path = None

    # Analyze a few frames (skip heavy loop for hackathon)
    cap = cv2.VideoCapture(video_path)
    results = []
    for _ in range(5):  # analyze first 5 frames only
        ret, frame = cap.read()
        if not ret:
            break
        res = video_detector.analyze_frame(frame)
        results.append(res)
    cap.release()

    # Process audio if extracted
    audio_result = None
    if audio_path:
        audio_result = await process_audio_input(audio_path)

    return {
        "type": "video_analysis",
        "result": {
            "frame_checks": results,
            "audio_analysis": audio_result
        }
    }

    if not video_detector:
        return {"type": "video_analysis", "result": {"error": "Video model is not available."}}

    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"type": "video_analysis", "result": {"face_detected": False, "error": "Invalid frame"}}

    deepfake_result = video_detector.analyze_frame(frame)
    return {"type": "video_analysis", "result": deepfake_result}