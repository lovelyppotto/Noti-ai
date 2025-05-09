import logging
from fastapi import APIRouter, HTTPException, status
import datetime
from pydantic import BaseModel, HttpUrl # 요청 본문 유효성 검사용

# Celery 작업 가져오기
# from app.services.celery_tasks import process_youtube_video_to_transcript # 직접 작업 함수 임포트
from services import celery_tasks # celery_tasks 모듈을 가져와서 사용

logger = logging.getLogger(__name__)
router = APIRouter()


class VideoProcessingRequest(BaseModel):
    url: HttpUrl # HttpUrl 타입으로 URL 유효성 검사
    summarize: bool = False # 요약 여부 (기본값 False)

@router.post("/stt/youtube-video", status_code=status.HTTP_202_ACCEPTED)
async def schedule_youtube_stt_task(
    req: VideoProcessingRequest,
):
    """
    YouTube 비디오 URL을 받아 STT 및 선택적 요약 작업을 Celery 큐에 등록합니다.
    작업 ID를 즉시 반환합니다.
    """
    logger.info(f"YouTube STT 작업 요청 수신: URL='{req.url}', 요약 여부={req.summarize}")

    try:
        # Celery 작업 호출
        # celery_tasks.py에 정의된 process_youtube_video_to_transcript 작업 함수를 호출합니다.
        task = celery_tasks.process_youtube_video_to_transcript.delay(
            url=str(req.url), # HttpUrl을 문자열로 변환하여 전달
            summarize=req.summarize
        )
        logger.info(f"Celery 작업이 성공적으로 등록되었습니다: 작업 ID='{task.id}', URL='{req.url}'")
        
        return {
            "message": "YouTube 비디오 STT 작업이 성공적으로 예약되었습니다.",
            "task_id": task.id,
            # 작업 상태를 확인할 수 있는 URL 예시 (실제 구현 필요)
            "status_check_url": f"/api/ai/tasks/status/{task.id}" 
        }
    except Exception as e:
        logger.error(f"Celery 작업 등록 중 오류 발생 (URL: {req.url}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="작업을 예약하는 동안 서버 내부 오류가 발생했습니다. 서버 로그를 확인해주세요."
        )
    
@router.get("/stt/youtube-video/{video_id}", status_code=status.HTTP_200_OK)
async def get_youtube_transcript(video_id: str):
    """
    YouTube 비디오 ID로 저장된 트랜스크립트를 가져옵니다.
    """
    logger.info(f"트랜스크립트 조회 요청: 비디오 ID='{video_id}'")
    
    try:
        # MongoDB에서 트랜스크립트 조회
        from models import transcript_collection
        transcript_doc = transcript_collection.find_one({"video_id": video_id})
        
        if not transcript_doc:
            logger.warning(f"요청한 트랜스크립트를 찾을 수 없습니다: {video_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"요청한 비디오 ID ({video_id})에 대한 트랜스크립트가 없습니다."
            )
        
        # MongoDB ObjectId를 문자열로 변환
        transcript_doc["_id"] = str(transcript_doc["_id"])
        # datetime을 Spring 호환 형식으로 변환
        for dt_field in ["created_at", "updated_at"]:
            if dt_field in transcript_doc and isinstance(transcript_doc[dt_field], datetime):
                transcript_doc[dt_field] = transcript_doc[dt_field].strftime('%Y-%m-%d %H:%M:%S.%f')
        
        return transcript_doc
        
    except HTTPException:
        raise  # 이미 생성된 HTTPException 그대로 전달
    except Exception as e:
        logger.error(f"트랜스크립트 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="트랜스크립트 조회 중 서버 내부 오류가 발생했습니다."
        )

@router.get("/tasks/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Celery 작업 상태를 확인합니다.
    """
    logger.info(f"작업 상태 확인 요청: 작업 ID='{task_id}'")
    
    try:
        from celery_config import celery_app
        task = celery_app.AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": task.status
        }
        
        # 추가 정보
        if task.status == 'SUCCESS':
            response["result"] = task.result
        elif task.status == 'FAILURE':
            response["error"] = str(task.result)
        elif task.status in ['PENDING', 'STARTED', 'RETRY']:
            response["info"] = "Task is in progress"
        
        return response
        
    except Exception as e:
        logger.error(f"작업 상태 확인 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="작업 상태를 확인하는 동안 서버 내부 오류가 발생했습니다."
        )
    
