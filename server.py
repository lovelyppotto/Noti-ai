import os
from collections import deque

import torch
import re
import numpy as np
from silero_vad import VADIterator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
)
from openai import OpenAI                         


OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)


device     = "cuda" if torch.cuda.is_available() else "cpu"
processor  = AutoProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer  = processor.tokenizer
model      = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

# VAD 모델 초기화
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)

# VAD 관련 함수
get_speech_timestamps, _, _, _, _ = utils
vad_model = vad_model.to(device)

# 무시할 단어 리스트
IGNORE_WORDS = ["음", "어", "음..", "어..", "그.."]


token_hist  = deque(maxlen=224)     # Whisper 프롬프트용 토큰 224개
MAX_CHARS   = 20_000                # 자막 전체 캐시(요약에 사용)
history_txt = ""                    # 실시간 누적 자막


def decode_audio(binary_audio: bytes) -> np.ndarray:
    """Float32 PCM 으로 변환"""
    return np.frombuffer(binary_audio, dtype=np.float32)

def is_speech(audio_np: np.ndarray, sample_rate=16000) -> bool:
    """VAD를 사용하여 음성인지 확인"""
    # NumPy 배열을 torch 텐서로 변환하고 올바른 장치로 이동
    audio_tensor = torch.tensor(audio_np).to(device)
    
    timestamps = get_speech_timestamps(
        audio_tensor, 
        vad_model, 
        sampling_rate=sample_rate,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )
    return len(timestamps) > 0

def filter_text(text: str) -> str:
    """무의미한 단어와 배경음악 표현 필터링"""
    # 짧은 무의미 단어 필터링
    for word in IGNORE_WORDS:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    
    # "배경음악", "음악" 등이 포함된 문장 필터링
    text = re.sub(r'[^\s]*(?:배경음악|음악)[^\s]*', '', text)
    
    # 2글자 미만의 단어만 있는 문장 필터링 (영어)
    text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def transcribe(audio_np: np.ndarray) -> str:
    """Whisper 전사 + 토큰 히스토리 유지 + 필터링"""
    global history_txt
    
    # VAD로 음성 여부 확인
    if not is_speech(audio_np):
        return ""  # 음성이 아니면 빈 문자열 반환
    
    # 개선된 프롬프트
    prompt = (tokenizer.decode(list(token_hist)) + 
             " 영어 고유명사 신경쓰기. 배경음악은 무시하고 사람 음성만 인식하세요.")

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
    
    # 텍스트 필터링
    filtered_text = filter_text(text)
    
    # 빈 결과면 처리하지 않음
    if not filtered_text:
        return ""
        
    # 토큰‧문자열 캐시 갱신
    token_hist.extend(tokenizer.encode(filtered_text, add_special_tokens=False))
    history_txt  = (history_txt + filtered_text + " ")[-MAX_CHARS:]
    return filtered_text.strip()

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
audio_buffer = deque()
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")
    try:
        while True:
            data = await ws.receive()

            if "bytes" in data:
                audio_np = decode_audio(data["bytes"])
                audio_buffer.append(audio_np)

                # 누적된 전체 오디오 샘플 수 계산
                total_samples = sum(chunk.shape[0] for chunk in audio_buffer)

                # 1초(16000 샘플) 이상 모였을 때만 인식
                if total_samples >= 16000:
                    # 버퍼 연결 (하나의 큰 배열로)
                    combined_audio = np.concatenate(list(audio_buffer))
                    audio_buffer.clear()

                    text = transcribe(combined_audio)
                    print(text)
                    await ws.send_text(text)

            elif "text" in data and data["text"] == "__SUMMARY__":
                summary = await summarize(history_txt)
                print("[요약]\n" + summary)
                await ws.send_text("[요약]\n" + summary)

    except WebSocketDisconnect:
        print("Client disconnected")
