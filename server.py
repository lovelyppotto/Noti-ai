from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import numpy as np
from collections import deque
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

app = FastAPI()

token_hist = deque(maxlen=224)          # O(224) 고정
MAX_CHARS = 2000                        # UI용 문자열 캐시 한도
history_txt = ""

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer    = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

def decode_audio(binary_audio):
    audio_np = np.frombuffer(binary_audio, dtype=np.float32)
    return audio_np

def transcribe(audio_np):
    global history_txt
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", language="ko", task="transcribe", prompt=tokenizer.decode(list(token_hist))+"영어 고유명사 신경쓰기")
    input_features = inputs.input_features.to(device)

    if input_features.dtype != torch.float16 and device == "cuda":
        input_features = input_features.half()

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

     # ① token deque 갱신
    token_hist.extend(tokenizer(transcription).input_ids)

    # ② UI 캐시 문자열 제한
    history_txt = (history_txt + transcription + " ")[-MAX_CHARS:]
    return transcription

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            message = await websocket.receive_bytes()
            try:
                audio_np = decode_audio(message)
                text = transcribe(audio_np)
                await websocket.send_text(text)
                print(f"Transcribed: {text}")
            except Exception as e:
                print(f"Error during transcription: {e}")
                await websocket.send_text(f"Error: {str(e)}")
    except WebSocketDisconnect:
        print("Client disconnected")
