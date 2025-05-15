import logging
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from celery import Celery
from fastapi.middleware.cors import CORSMiddleware

# 설정 로드 (가장 먼저 실행되도록)
from config import settings

# 라우터 임포트
from api.websocket import router as websocket_router
from api.backgroundstt import router as background_stt_router

# 로깅 기본 설정
logging.basicConfig(level=settings.LOG_LEVEL.upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작 및 종료 시 실행될 로직을 관리합니다.
    """
    logger.info("애플리케이션 시작...")

    try:
        from services.whisper_service import get_whisper_model_instance
        model_instance = get_whisper_model_instance() # 모델 인스턴스 가져오기
        if model_instance:
            logger.info("Whisper 모델이 성공적으로 로드되었습니다.")
        else:
            logger.error("Whisper 모델 로드에 실패했습니다.")
    except Exception as e:
        logger.error(f"Whisper 모델 로드 중 오류 발생: {e}", exc_info=True)
    
    yield  # 애플리케이션 실행 구간

    # 종료 시 리소스 정리
    logger.info("애플리케이션 종료 중...")
    if settings.WHISPER_DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA 캐시가 정리되었습니다.")
        except Exception as e:
            logger.error(f"CUDA 캐시 정리 중 오류 발생: {e}", exc_info=True)
    logger.info("애플리케이션이 성공적으로 종료되었습니다.")

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="Background STT and OCR service",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    
        "https://notii.kr",   
    ],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],   
)


# API 라우터 등록
app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
app.include_router(background_stt_router, prefix="/api/ai", tags=["Background STT"])


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "STT 및 요약 서비스에 오신 것을 환영합니다!"}

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Uvicorn 서버를 시작합니다 (개발용).")
#     uvicorn.run(app, host="0.0.0.0", port=8000)