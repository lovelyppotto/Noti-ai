# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# import torch
# import numpy as np
# from collections import deque
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# app = FastAPI()

# token_hist = deque(maxlen=224)          # O(224) 고정
# MAX_CHARS = 2000                        # UI용 문자열 캐시 한도
# history_txt = ""

# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
# tokenizer    = processor.tokenizer
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     "openai/whisper-large-v3",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     low_cpu_mem_usage=True
# ).to(device)

# def decode_audio(binary_audio):
#     audio_np = np.frombuffer(binary_audio, dtype=np.float32)
#     return audio_np

# def transcribe(audio_np):
#     global history_txt
#     inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", language="ko", task="transcribe", prompt=tokenizer.decode(list(token_hist))+"영어 고유명사 신경쓰기")
#     input_features = inputs.input_features.to(device)

#     if input_features.dtype != torch.float16 and device == "cuda":
#         input_features = input_features.half()

#     predicted_ids = model.generate(input_features)
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

#      # ① token deque 갱신
#     token_hist.extend(tokenizer(transcription).input_ids)

#     # ② UI 캐시 문자열 제한
#     history_txt = (history_txt + transcription + " ")[-MAX_CHARS:]
#     return transcription

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Client connected")
#     try:
#         while True:
#             message = await websocket.receive_bytes()
#             try:
#                 audio_np = decode_audio(message)
#                 text = transcribe(audio_np)
#                 await websocket.send_text(text)
#                 print(f"Transcribed: {text}")
#             except Exception as e:
#                 print(f"Error during transcription: {e}")
#                 await websocket.send_text(f"Error: {str(e)}")
#     except WebSocketDisconnect:
#         print("Client disconnected")
# server.py  ---------------------------------------------------------
import os
from collections import deque

import torch
import numpy as np
from silero_vad import VADIterator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
)
from openai import OpenAI                           # ↙ GPT-4.1-mini

# -------------------------------------------------------------------
# 0. 환경 변수에서 OpenAI 키 읽기
#    (터미널:  export OPENAI_API_KEY="sk-xxxx")
# -------------------------------------------------------------------
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# 1. Whisper 모델‧토크나이저 준비
# -------------------------------------------------------------------
device     = "cuda" if torch.cuda.is_available() else "cpu"
processor  = AutoProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer  = processor.tokenizer
model      = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

# -------------------------------------------------------------------
# 2. 상태 캐시
# -------------------------------------------------------------------
token_hist  = deque(maxlen=224)     # Whisper 프롬프트용 토큰 224개
MAX_CHARS   = 20_000                # 자막 전체 캐시(요약에 사용)
history_txt = ""                    # 실시간 누적 자막
vad = VADIterator(sample_rate=16000)

# -------------------------------------------------------------------
# 3. 보조 함수
# -------------------------------------------------------------------
def decode_audio(binary_audio: bytes) -> np.ndarray:
    """Float32 PCM 으로 변환"""
    return np.frombuffer(binary_audio, dtype=np.float32)

def transcribe(audio_np: np.ndarray) -> str:
    """Whisper 전사 + 토큰 히스토리 유지"""
    global history_txt
    prompt = tokenizer.decode(list(token_hist)) + " 영어 고유명사 신경쓰기"

    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        language="ko",
        task="transcribe",
        prompt=prompt,
    )
    feats = inputs.input_features.to(device)
    if feats.dtype != torch.float16 and device == "cuda":
        feats = feats.half()

    pred_ids = model.generate(feats)
    text     = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    # 토큰‧문자열 캐시 갱신
    token_hist.extend(tokenizer.encode(text, add_special_tokens=False))
    history_txt  = (history_txt + text + " ")[-MAX_CHARS:]
    return text.strip()

async def summarize(full_text: str) -> str:
    """GPT-4.1-mini 한글 요약 (5줄 이내 불릿)"""
    system_msg = (
        "당신은 한국어로된 강의 자막을 핵심 위주로 요약하는 AI 비서입니다. "
        "핵심 키워드를 보존하되 소제목과 불릿으로 정리하세요."
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": full_text},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------------------------------------
# 4. FastAPI WebSocket 엔드포인트
# -------------------------------------------------------------------
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")
    try:
        while True:
            data = await ws.receive()

            # ① 오디오 조각
            if "bytes" in data:
                audio_np = decode_audio(data["bytes"])
                text     = transcribe(audio_np)
                print(text)
                await ws.send_text(text)

            # ② 요약 요청
            elif "text" in data and data["text"] == "__SUMMARY__":
                summary = await summarize(history_txt)
                await ws.send_text("[요약]\n" + summary)

    except WebSocketDisconnect:
        print("Client disconnected")
