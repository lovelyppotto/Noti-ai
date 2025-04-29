from fastapi import FastAPI
from app.api.websocket import router as websocket_router

app = FastAPI()

# WebSocket 라우터 등록
app.include_router(websocket_router)