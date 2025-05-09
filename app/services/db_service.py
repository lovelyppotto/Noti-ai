import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from models import transcript_collection
from services.youtube_service import extract_video_id

logger = logging.getLogger(__name__)

async def find_transcript_by_video_id(video_id: str) -> Optional[Dict[str, Any]]:
    """
    비디오 ID로 트랜스크립트를 찾습니다.
    """
    try:
        result = await transcript_collection.find_one({"video_id": video_id})
        return result
    except Exception as e:
        logger.error(f"트랜스크립트 검색 중 오류 발생: {e}", exc_info=True)
        return None

async def find_transcript_by_url(url: str) -> Optional[Dict[str, Any]]:
    """
    URL로 트랜스크립트를 찾습니다.
    """
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"유효하지 않은 YouTube URL: {url}")
        return None
    
    return await find_transcript_by_video_id(video_id)

async def save_transcript(
    video_id: str,
    url: str,
    title: str,
    transcript_segments: List[Dict[str, Any]],
    full_text: str,
    summary: Optional[str] = None
) -> Optional[str]:
    """
    트랜스크립트를 데이터베이스에 저장합니다.
    이미 존재하는 경우 업데이트합니다.
    """
    try:
        now = datetime.utcnow()
        document = {
            "video_id": video_id,
            "url": url,
            "title": title,
            "transcript": transcript_segments,
            "full_text": full_text,
            "summary": summary,
            "updated_at": now
        }
        
        # 이미 존재하는 경우 업데이트, 없으면 새로 생성
        result = await transcript_collection.update_one(
            {"video_id": video_id},
            {"$set": document, "$setOnInsert": {"created_at": now}},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"새 트랜스크립트 생성됨: {video_id}")
            return str(result.upserted_id)
        else:
            logger.info(f"기존 트랜스크립트 업데이트됨: {video_id}")
            return video_id
            
    except Exception as e:
        logger.error(f"트랜스크립트 저장 중 오류 발생: {e}", exc_info=True)
        return None