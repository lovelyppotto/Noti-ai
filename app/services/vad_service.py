import torch
import numpy as np
from collections import deque

# VAD 모델 초기화
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)
get_speech_timestamps, _, _, _, _ = utils
device = "cuda" if torch.cuda.is_available() else "cpu"
vad_model = vad_model.to(device)

audio_buffer = deque()

def decode_audio(binary_audio: bytes) -> np.ndarray:
    """Float32 PCM 으로 변환"""
    return np.frombuffer(binary_audio, dtype=np.float32)

def is_speech(audio_np: np.ndarray, sample_rate=16000) -> bool:
    """VAD를 사용하여 음성인지 확인"""
    audio_tensor = torch.tensor(audio_np).to(device)
    timestamps = get_speech_timestamps(
        audio_tensor, 
        vad_model, 
        sampling_rate=sample_rate,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )
    return len(timestamps) > 0
