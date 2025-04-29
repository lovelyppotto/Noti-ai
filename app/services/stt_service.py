import soundfile as sf
from app.services.whisper_service import transcribe
from app.services.summary_service import summarize

async def transcribe_video(audio_path: str, do_summary: bool) -> str:
    # ① 오디오 전체 로드 (16 kHz mono)
    audio_np, sr = sf.read(audio_path)
    if sr != 16000:
        raise RuntimeError("WAV 샘플레이트가 16 kHz 가 아닙니다.")

    # ② 30 s(480k)씩 슬라이딩 중첩(5 s) 전사
    chunk_size   = 16000 * 30
    overlap      = 16000 * 5
    transcript   = []

    pos = 0
    while pos < len(audio_np):
        chunk = audio_np[pos : pos + chunk_size]
        text  = transcribe(chunk)
        if text:
            transcript.append(text)
        pos += chunk_size - overlap

    full_text = " ".join(transcript)

    # ③ 필요 시 요약
    if do_summary:
        summary = await summarize(full_text)
        return "[요약]\n" + summary + "\n\n[전체 자막]\n" + full_text
    return full_text
