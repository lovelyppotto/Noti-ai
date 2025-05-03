import torch
from collections import deque
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from app.services.vad_service import is_speech
from app.utils.filters import filter_text

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

token_hist = deque(maxlen=224)
MAX_CHARS = 20_000
history_txt = ""  # 전역적으로 사용

def transcribe(audio_np):
    global history_txt
    if not is_speech(audio_np):
        return ""
    prompt = (" 영어 고유명사 신경쓰기. 배경음악은 무시하고 사람 음성만 인식하세요." +tokenizer.decode(list(token_hist)))

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
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    filtered_text = filter_text(text)
    if not filtered_text:
        return ""
    token_hist.extend(tokenizer.encode(filtered_text, add_special_tokens=False))
    history_txt = (history_txt + filtered_text + " ")[-MAX_CHARS:]
    return filtered_text.strip()
