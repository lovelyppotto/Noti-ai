from fastapi import FastAPI
from app.api.websocket import router as websocket_router
from app.api.backgroundstt import router as background_stt_router

app = FastAPI()

# WebSocket 라우터 등록
app.include_router(websocket_router)
app.include_router(background_stt_router)