# Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential libffi-dev \
    libsndfile1 ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 애플리케이션 코드 복사
COPY . .

# 웹 서비스 포트 노출
EXPOSE 8000

# 기본 명령 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]