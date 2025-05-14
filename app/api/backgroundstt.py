import logging
from fastapi import APIRouter, HTTPException, status
import datetime
from pydantic import BaseModel, HttpUrl # 요청 본문 유효성 검사용
from models import transcript_collection, async_transcript_collection
from services.transcript_service import YouTubeTranscriptService
from services.youtube_service import YouTubeProcessor
from services.db_service import save_transcript

from pydantic import BaseModel, HttpUrl
from services.ocr_service import ocr_service

# Celery 작업 가져오기
# from app.services.celery_tasks import process_youtube_video_to_transcript # 직접 작업 함수 임포트
from services import celery_tasks # celery_tasks 모듈을 가져와서 사용

logger = logging.getLogger(__name__)
router = APIRouter()
youtube_processor = YouTubeProcessor()

class OCRRequest(BaseModel):
    s3_url: str  # S3 이미지 URL
 
 
class VideoProcessingRequest(BaseModel):
    url: HttpUrl # HttpUrl 타입으로 URL 유효성 검사
    summarize: bool = False # 요약 여부 (기본값 False)

class YouTubeTranscriptCheckRequest(BaseModel):
    url: HttpUrl

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
    
@router.get("/youtube/script/check", status_code=status.HTTP_200_OK)
async def check_youtube_transcript(url: str):
    """
    YouTube 영상에 한국어 자막이 존재하는지 확인합니다.
    """
    logger.info(f"YouTube 한국어 자막 확인 요청: URL='{url}'")
    
    try:
        # URL에서 비디오 ID 추출
        video_id = youtube_processor.extract_video_id(str(url))
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 YouTube URL입니다."
            )
        
        
        # 한국어 자막 존재 여부 확인
        transcript_service = YouTubeTranscriptService()
        has_korean = transcript_service.check_korean_transcript(video_id)
        
        return {
            "video_id": video_id,
            "has_transcript": has_korean,
            "in_database": False,
            "message": "한국어 자막이 있습니다." if has_korean else "한국어 자막이 없습니다."
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube 자막 확인 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YouTube 자막 확인 중 서버 내부 오류가 발생했습니다."
        )
    
@router.post("/youtube/script/save", status_code=status.HTTP_201_CREATED)
async def save_youtube_transcript(req: YouTubeTranscriptCheckRequest):
    """
    YouTube 영상의 한국어 자막을 가져와 데이터베이스에 저장합니다.
    """
    logger.info(f"YouTube 한국어 자막 저장 요청: URL='{req.url}'")
    
    try:
        # URL에서 비디오 ID 추출
        video_id = youtube_processor.extract_video_id(str(req.url))
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 YouTube URL입니다."
            )
        
        # 이미 존재하는지 확인
        existing_doc = await async_transcript_collection.find_one({"video_id": video_id})
        if existing_doc:
            return {
                "success": True,
                "video_id": video_id,
                "message": "해당 비디오의 자막이 이미 저장되어 있습니다.",
                "in_database": True
            }
        
        # YouTube 비디오 정보 가져오기
        video_info = youtube_processor.get_video_info(str(req.url))
        
        if not video_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="YouTube 비디오 정보를 찾을 수 없습니다."
            )
        
        # 한국어 자막 존재 여부 확인
        transcript_service = YouTubeTranscriptService()
        has_korean = transcript_service.check_korean_transcript(video_id)
        
        if not has_korean:
            return {
                "success": False,
                "video_id": video_id,
                "message": "한국어 자막이 없습니다.",
                "in_database": False
            }
        
        # 한국어 자막 가져오기
        transcript_data = transcript_service.get_korean_transcript(video_id)
        
        if not transcript_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="한국어 자막을 찾을 수 없습니다."
            )
        
        # 자막 데이터를 기존 형식에 맞게 변환
        transcript_segments = []
        full_text = ""
        transcript_snippets = transcript_data.snippets
        for snippet in transcript_snippets:
            # 속성으로 직접 접근 (snippet.start, snippet.duration, snippet.text)
            if hasattr(snippet, 'start') and hasattr(snippet, 'duration') and hasattr(snippet, 'text'):
                # FetchedTranscriptSnippet 객체인 경우
                start = snippet.start
                duration = snippet.duration
                text = snippet.text
            elif isinstance(snippet, dict):
                # 딕셔너리인 경우 (이전 버전 호환성)
                start = snippet.get('start', 0.0)
                duration = snippet.get('duration', 0.0)
                text = snippet.get('text', '')
            else:
                # 알 수 없는 형식은 스킵
                logger.warning(f"알 수 없는 스니펫 형식: {type(snippet)}")
                continue
            
            end = start + duration
            
            # 세그먼트 추가
            transcript_segments.append({
                "start": start,
                "end": end,
                "text": text
            })
            
            # 전체 텍스트 구성
            full_text += text + " "
        
        if not transcript_segments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="유효한 자막 세그먼트를 추출할 수 없습니다."
            )
        
        # 트랜스크립트 저장
        document_id = await save_transcript(
            video_id=video_id,
            url=str(req.url),
            title=video_info.get("title", ""),
            transcript_segments=transcript_segments,
            full_text=full_text.strip(),
            summary=None
        )
        
        if document_id:
            return {
                "success": True,
                "video_id": video_id,
                "document_id": document_id,
                "title": video_info.get("title", ""),
                "segments_count": len(transcript_segments),
                "message": "YouTube 자막이 성공적으로 저장되었습니다."
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="문서 삽입에 실패했습니다."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube 자막 저장 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YouTube 자막 저장 중 서버 내부 오류가 발생했습니다."
        )
        
@router.post("/ocr", status_code=status.HTTP_200_OK)
async def process_ocr_request(req: OCRRequest):
    """
    이미지에서 텍스트를 추출하는 동기식 OCR 엔드포인트
    
    Args:
        req: S3 URL이 포함된 요청
        
    Returns:
        Dict: OCR 처리 결과
    """
    logger.info(f"OCR 처리 요청 수신: S3 URL='{req.s3_url}'")
    
    try:
        result = ocr_service.process_image(s3_url=req.s3_url)
        
        # 처리 실패 시 예외 발생
        if not result.get("success"):
            logger.warning(f"OCR 처리 실패: {result.get('message')}")
            return {
                "code": 200,
                "message": "OCR 처리 중 오류가 발생했으나, 빈 텍스트를 반환합니다.",
                "text": "텍스트를 추출할 수 없습니다.",
                "text_items": []
            }
        
        # None 값을 확인하고 빈 리스트로 대체
        text_items = result.get("text_items", [])
        if text_items is None:
            text_items = []
        
        # None 값이나 빈 문자열 항목 필터링
        safe_text_items = []
        for item in text_items:
            if item is not None and item.strip():
                safe_text_items.append(item)
        
        # 안전한 로깅
        item_count = len(safe_text_items) if safe_text_items is not None else 0
        logger.info(f"OCR 처리 완료: URL='{req.s3_url}', 텍스트 항목 수={item_count}")
        
        # 빈 텍스트인 경우 기본값 제공
        full_text = result.get("full_text", "").strip()
        if not full_text:
            # 텍스트 항목이 있으면 합쳐서 사용
            if safe_text_items:
                full_text = " ".join(safe_text_items)
            else:
                full_text = "텍스트를 추출할 수 없습니다."
        
        return {
            "code": 200,
            "message": "OCR 작업이 성공적으로 완료되었습니다.",
            "text": full_text,
            "text_items": safe_text_items
        }

    except Exception as e:
        logger.error(f"OCR 처리 중 예외 발생: {e}", exc_info=True)
        
        # 오류 발생 시 빈 텍스트 반환
        return {
            "code": 200,
            "message": "OCR 처리 중 오류가 발생했으나, 빈 텍스트를 반환합니다.",
            "text": "",
            "text_items": []
        }