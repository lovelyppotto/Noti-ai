import os
import subprocess
import tempfile
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from app.services.stt_service import transcribe_video

router = APIRouter()

# ---------- 요청 스키마 ----------
class VideoReq(BaseModel):
    url: HttpUrl
    summarize: bool = False

# ---------- 엔드포인트 ----------
@router.post("/ai/stt")
async def stt_video(req: VideoReq, background_tasks: BackgroundTasks):
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")

        # yt-dlp: 오디오만 추출, 16 kHz mono
        cmd = [
            "yt-dlp", "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "-ar 16000 -ac 1",
            "-o", wav_path,
            req.url
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(400, "YouTube 다운로드 실패:\n" + proc.stderr[-500:])

        # 비동기 백그라운드 전사 → 즉시 202 응답
        background_tasks.add_task(transcribe_video, wav_path, req.summarize)
        return {"detail": "Transcription started. Check logs or websocket for result."}
