# ui/app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF logs (0 = all, 3 = only errors)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from alerts.alert_manager import ConnectionManager
from pipeline.detection_pipeline import (
    process_text_input, 
    process_audio_input, 
    process_video_file
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mount static files using an absolute path
app.mount("/static", StaticFiles(directory=os.path.join(project_dir, "ui/static")), name="static")

manager = ConnectionManager()

@app.get("/")
async def get():
    template_path = os.path.join(project_dir, "ui/templates/index.html")
    with open(template_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/analyze/text")
async def analyze_text_endpoint(data: dict):
    text_content = data.get("text")
    if not text_content:
        return {"error": "No text provided"}, 400
    
    result = await process_text_input(text_content)
    await manager.broadcast(result)
    return {"status": "Text analysis triggered", "details": result}

@app.post("/analyze/audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    temp_audio_path = f"temp_{file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        buffer.write(await file.read())
    
    result = await process_audio_input(temp_audio_path)
    os.remove(temp_audio_path)
    
    await manager.broadcast(result)
    return {"status": "Audio analysis triggered", "details": result}

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            temp_video_path = "temp_ws_video_frame.mp4"
            with open(temp_video_path, "wb") as temp_file:
                temp_file.write(frame_bytes)
            result = await process_video_file(temp_video_path)
            os.remove(temp_video_path)
            
            deepfake_analysis = result.get("result", {})
            if deepfake_analysis.get("face_detected") and not deepfake_analysis.get("is_real", True):
                await manager.broadcast(result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from video stream.")

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from alerts.")

@app.post("/analyze/video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())

    result = await process_video_file(temp_video_path)
    os.remove(temp_video_path)

    await manager.broadcast(result)
    return {"status": "Video analysis triggered", "details": result}
