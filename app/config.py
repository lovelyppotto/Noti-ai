import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List, Optional, Dict
from urllib import parse

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    # Redis 기본 설정
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: str = os.environ.get("REDIS_PORT", "6379")
    REDIS_USERNAME: str = os.environ.get("REDIS_USERNAME", "default")
    REDIS_PASSWORD: str = os.environ.get("REDIS_PASSWORD", "")  # Redis 비밀번호
    REDIS_DB_BROKER: str = os.environ.get("REDIS_DB_BROKER", "0")
    REDIS_DB_BACKEND: str = os.environ.get("REDIS_DB_BACKEND", "1")
    
    # Redis URL 생성 (비밀번호가 있는 경우)
    @property
    def CELERY_BROKER_URL(self) -> str:
        if self.REDIS_PASSWORD:
            # 비밀번호에 특수 문자가 있을 수 있으므로 URL 인코딩
            encoded_password = parse.quote_plus(self.REDIS_PASSWORD)
            return f"redis://{self.REDIS_USERNAME}:{encoded_password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_BROKER}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_BROKER}"
    
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        if self.REDIS_PASSWORD:
            encoded_password = parse.quote_plus(self.REDIS_PASSWORD)
            return f"redis://{self.REDIS_USERNAME}:{encoded_password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_BACKEND}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_BACKEND}"
    
    CELERY_TASK_TIME_LIMIT: int = 3600  # 작업 시간 제한 (초)
    CELERY_WORKER_CONCURRENCY: int = 2   # 워커 동시성

    # OpenAI API 설정
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_SUMMARY_MODEL: str = "gpt-4.1" 
    OPENAI_SUMMARY_MAX_TOKENS: int = 1024
    OPENAI_SUMMARY_TEMPERATURE: float = 0.2

    # Whisper 모델 설정
    WHISPER_MODEL_NAME: str = "openai/whisper-large-v3"
    WHISPER_DEVICE: str = "cuda" # "cuda" or "cpu"
    WHISPER_TORCH_DTYPE: str = "float16" # "float16", "float32", "bfloat16" (상황에 맞게)
    WHISPER_LOW_CPU_MEM_USAGE: bool = True
    WHISPER_USE_SAFETENSORS: bool = True
    WHISPER_TOKEN_HISTORY_MAXLEN: int = 224 # Whisper 프롬프트에 사용될 토큰 히스토리 최대 길이
    WHISPER_MAX_CHARS_HISTORY: int = 20000 # 요약 시 사용될 전체 텍스트 히스토리 최대 문자 수
    WHISPER_SAMPLING_RATE: int = 16000
    WHISPER_CHUNK_SIZE_SECONDS: int = 30 # 긴 오디오 처리 시 청크 크기 (초)

    # VAD 모델 설정
    VAD_MODEL_REPO_OR_DIR: str = 'snakers4/silero-vad'
    VAD_MODEL_NAME: str = 'silero_vad'
    VAD_FORCE_RELOAD: bool = False # 프로덕션에서는 False 권장
    VAD_MIN_SPEECH_DURATION_MS: int = 250
    VAD_MIN_SILENCE_DURATION_MS: int = 100
    VAD_SPEECH_PAD_MS: int = 30 # 음성 앞뒤로 추가할 패딩 (ms)

    # MongoDB 설정
    MONGODB_URI: str = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.environ.get("MONGODB_DB_NAME", "noti")
    MONGODB_TRANSCRIPT_COLLECTION: str = "video_transcripts"

    # 로깅 설정
    LOG_LEVEL: str = "INFO"

    # 필터링 설정
    FILTER_IGNORE_WORDS: List[str] = ["음", "어", "음...", "어...", "그...", "저...", "아..."]
    FILTER_ALLOWED_SHORT_ENGLISH_WORDS: List[str] = ["AI", "IT", "ML", "DL", "UI", "UX", "API", "OK"]

    # Webshare Proxy Settings
    PROXY_URL_WEBSHARE: Optional[str] = os.environ.get("PROXY_URL_WEBSHARE")
    HTTP_PROXY_WEBSHARE: Optional[str] = os.environ.get("HTTP_PROXY_WEBSHARE")
    HTTPS_PROXY_WEBSHARE: Optional[str] = os.environ.get("HTTPS_PROXY_WEBSHARE")

    # youtube-transcript-api에 전달할 프록시 딕셔너리 생성
    @property
    def TRANSCRIPT_API_PROXIES_WEBSHARE(self) -> Optional[Dict[str, str]]:
        proxies = {}
        if self.HTTP_PROXY_WEBSHARE:
            proxies['http'] = self.HTTP_PROXY_WEBSHARE
        if self.HTTPS_PROXY_WEBSHARE:
            proxies['https'] = self.HTTPS_PROXY_WEBSHARE
        return proxies if proxies else None

    # pydantic에서 자동으로 .env 파일을 로드하도록 설정
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 설정 객체 인스턴스화
settings = Settings()