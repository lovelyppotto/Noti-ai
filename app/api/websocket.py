from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.vad_service import decode_audio, audio_buffer
from app.services.whisper_service import transcribe, history_txt
from app.services.summary_service import summarize

router = APIRouter()

@router.websocket("/ws/realtime")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")
    try:
        while True:
            data = await ws.receive()

            if "bytes" in data:
                audio_np = decode_audio(data["bytes"])
                audio_buffer.append(audio_np)

                total_samples = sum(chunk.shape[0] for chunk in audio_buffer)
                if total_samples >= 16000:
                    combined_audio = audio_buffer[0] if len(audio_buffer) == 1 else __import__('numpy').concatenate(list(audio_buffer))
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
