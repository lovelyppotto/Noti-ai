from celery import Celery
from config import settings # 기존 설정 파일
import redis
import sys

# Celery 앱 인스턴스 생성
celery_app = Celery(
    'app', # 애플리케이션 이름 (Celery 4.x 이상에서는 필수는 아님)
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['services.celery_tasks']
)

# Windows에서 실행 중인지 확인
is_windows = sys.platform.startswith('win')

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    worker_concurrency=1 if is_windows else settings.CELERY_WORKER_CONCURRENCY,
    task_acks_late=True,
    worker_prefetch_multiplier=1, # 긴 작업에 유리할 수 있도록 설정
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,

    # Redis 연결 풀 설정 추가
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
)

# Windows에서 브로커 URL 확인 (추가 검증)
if is_windows:
    # Redis 서버 연결 테스트
    try:
        import redis
        redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        redis_client.ping()
        print("Redis 서버 연결 성공")
    except Exception as e:
        print(f"Redis 서버 연결 실패: {e}")

if __name__ == '__main__':
    # Celery 워커를 직접 실행하기 위한 스크립트 (예시)
    # 실제로는 터미널에서 `celery -A app.core.celery_config.celery_app worker -l info` 명령 사용
    celery_app.start()
