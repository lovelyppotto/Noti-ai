# services/transcript_service.py

import logging
from typing import List, Dict, Any, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from config import settings

logger = logging.getLogger(__name__)

class YouTubeTranscriptService:
    """YouTube 자막 관련 기능을 제공하는 서비스 클래스"""
    @staticmethod
    def _get_proxies_for_transcript_api() -> Optional[Dict[str, str]]:
        return settings.TRANSCRIPT_API_PROXIES_WEBSHARE
    

    @staticmethod
    def check_korean_transcript(video_id: str) -> bool:
        proxies = YouTubeTranscriptService._get_proxies_for_transcript_api()
        print(settings.PROXY_URL_WEBSHARE)
        print(settings.HTTP_PROXY_WEBSHARE)
        print(settings.HTTPS_PROXY_WEBSHARE)
        log_proxy_info = f"(프록시: {proxies})" if proxies else "(프록시 사용 안함)"

        try:
            logger.debug(f"한국어 자막 확인 시도 (비디오 ID: {video_id}) {log_proxy_info}")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies)
            for transcript in transcript_list:
                if transcript.language_code == 'ko':
                    logger.info(f"한국어 자막 찾음 (비디오 ID: {video_id})")
                    return True
            logger.info(f"한국어 자막 없음 (비디오 ID: {video_id})")
            return False
        except TranscriptsDisabled:
            logger.warning(f"자막이 비활성화된 비디오 (비디오 ID: {video_id}) {log_proxy_info}")
            return False
        except NoTranscriptFound:
            logger.info(f"어떤 언어의 자막도 찾을 수 없음 (비디오 ID: {video_id}) {log_proxy_info}")
            return False
        except Exception as e:
            logger.error(f"한국어 자막 확인 중 오류 발생 (비디오 ID: {video_id}) {log_proxy_info}: {e}", exc_info=True)
            return False

    @staticmethod
    def get_korean_transcript(video_id: str) -> List[Dict[str, Any]]:
        proxies = YouTubeTranscriptService._get_proxies_for_transcript_api()
        log_proxy_info = f"(프록시: {proxies})" if proxies else "(프록시 사용 안함)"

        try:
            logger.debug(f"한국어 자막 가져오기 시도 (비디오 ID: {video_id}) {log_proxy_info}")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies)
            transcript = transcript_list.find_transcript(['ko'])
            fetched_transcript = transcript.fetch()
            logger.info(f"한국어 자막 성공적으로 가져옴 (비디오 ID: {video_id}), 항목 수: {len(fetched_transcript)}")
            return fetched_transcript
        except TranscriptsDisabled:
            logger.warning(f"자막이 비활성화된 비디오 (비디오 ID: {video_id}) {log_proxy_info}")
            return []
        except NoTranscriptFound:
            logger.info(f"한국어 자막을 찾을 수 없음 (비디오 ID: {video_id}) {log_proxy_info}")
            return []
        except Exception as e:
            logger.error(f"한국어 자막 가져오기 중 오류 발생 (비디오 ID: {video_id}) {log_proxy_info}: {e}", exc_info=True)
            return []