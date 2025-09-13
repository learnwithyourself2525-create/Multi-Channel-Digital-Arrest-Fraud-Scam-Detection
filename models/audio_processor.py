import whisper
import torchaudio
import torch
import os

class AudioProcessor:
    def __init__(self):
        print("INFO:     Initializing AudioProcessor (Whisper without FFmpeg)...")
        self.model = whisper.load_model("small")

    def process_audio(self, audio_path: str):
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found."}
        try:
            # Load audio with torchaudio instead of ffmpeg
            waveform, sr = torchaudio.load(audio_path)

            # Whisper expects 16000 Hz mono
            if waveform.shape[0] > 1:  # stereo → take mean to make mono
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            audio = waveform.squeeze().numpy()

            # Transcribe
            result = self.model.transcribe(audio, fp16=torch.cuda.is_available())
            transcribed_text = " ".join(result["text"]).strip() if isinstance(result["text"], list) else result["text"].strip()

            return {"transcribed_text": transcribed_text}
        except Exception as e:
            return {"error": f"Failed to process audio: {str(e)}"}
