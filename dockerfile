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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install paddlepaddle-gpu==2.6.2 -i https://pypi.org/simple

# 애플리케이션 코드 복사
COPY . .

# 웹 서비스 포트 노출
EXPOSE 8000

# 기본 명령 설정
CMD ["sh", "-c", "cd app && uvicorn main:app --host 0.0.0.0 --port 8000"]