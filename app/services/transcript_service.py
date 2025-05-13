# services/transcript_service.py

import logging
from typing import List, Dict, Any, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

logger = logging.getLogger(__name__)

class YouTubeTranscriptService:
    """YouTube 자막 관련 기능을 제공하는 서비스 클래스"""
    
    @staticmethod
    def check_korean_transcript(video_id: str) -> bool:

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # 한국어 자막 찾기 (언어 코드 'ko')
            for transcript in transcript_list:
                if transcript.language_code == 'ko':
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"한국어 자막 확인 실패 (비디오 ID: {video_id}): {e}", exc_info=True)
            return False
    
    @staticmethod
    def get_korean_transcript(video_id: str) -> List[Dict[str, Any]]:
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # 한국어 자막 찾기 (언어 코드 'ko')
            transcript = transcript_list.find_transcript(['ko'])
            return transcript.fetch()
                
        except Exception as e:
            logger.error(f"한국어 자막 가져오기 실패 (비디오 ID: {video_id}): {e}", exc_info=True)
            return []