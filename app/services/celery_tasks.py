import logging
import os
import tempfile
from datetime import datetime
# celery_config에서 celery_app 임포트
from celery_config import celery_app
from services.youtube_service import YouTubeProcessor
from services.whisper_service import TranscriptionService # STT 서비스 클래스 사용

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_youtube_video_to_transcript(self, url: str, summarize: bool = False):
    """
    YouTube URL에서 오디오를 다운로드하고 STT를 수행하는 Celery 작업.
    선택적으로 요약도 수행합니다.
    """
    logger.info(f"[Task ID: {self.request.id}] YouTube 처리 작업 시작: {url}, 요약: {summarize}")
    
    youtube_processor = YouTubeProcessor()
    transcription_service = TranscriptionService()
    
    # MongoDB 컬렉션을 직접 설정
    from pymongo import MongoClient
    from config import settings
    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]
    transcript_collection = db[settings.MONGODB_TRANSCRIPT_COLLECTION]
    
    # 비디오 ID 추출
    video_id = youtube_processor.extract_video_id(url)
    
    if not video_id:
        logger.error(f"[Task ID: {self.request.id}] 유효하지 않은 YouTube URL: {url}")
        self.update_state(state='FAILURE', meta={'exc_type': 'InvalidURLError', 'exc_message': '유효하지 않은 YouTube URL입니다.'})
        raise ValueError("Invalid YouTube URL")
    
    # 기존 트랜스크립트가 있는지 확인
    existing_transcript = transcript_collection.find_one({"video_id": video_id})
    
    if existing_transcript:
        logger.info(f"[Task ID: {self.request.id}] 이미 존재하는 트랜스크립트 발견: {video_id}")
        
        # 요약만 수행할 경우
        if summarize and not existing_transcript.get('summary'):
            if existing_transcript.get('full_text'):
                try:
                    from services.summary_service import summarize_text_sync
                    logger.info(f"[Task ID: {self.request.id}] 기존 트랜스크립트에 대한 요약 작업 시작...")
                    summary = summarize_text_sync(existing_transcript['full_text'])
                    
                    # 요약 업데이트
                    transcript_collection.update_one(
                        {"video_id": video_id},
                        {"$set": {"summary": summary, "updated_at": datetime.utcnow()}}
                    )
                    logger.info(f"[Task ID: {self.request.id}] 요약 추가 완료.")
                except Exception as e:
                    logger.error(f"[Task ID: {self.request.id}] 요약 중 오류 발생: {e}", exc_info=True)
        
        # 기존 트랜스크립트 반환
        result = {
            "transcript": existing_transcript.get('full_text', ''),
            "segments": existing_transcript.get('transcript', []),
            "summary": existing_transcript.get('summary'),
            "video_id": video_id,
            "title": existing_transcript.get('title', ''),
            "url": url,
            "status": "existing"
        }
        return result

    # 기존 트랜스크립트가 없으면 새로 처리
    if not all([transcription_service.model, transcription_service.processor, transcription_service.tokenizer]):
        logger.error(f"[Task ID: {self.request.id}] STT 서비스 초기화 실패. 작업 중단.")
        self.update_state(state='FAILURE', meta={'exc_type': 'InitializationError', 'exc_message': 'STT service failed to initialize.'})
        raise Exception("STT service initialization failed.")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = None
        try:
            # 비디오 정보 가져오기
            video_info = youtube_processor.get_video_info(url)
            if not video_info:
                logger.error(f"[Task ID: {self.request.id}] 비디오 정보를 가져올 수 없습니다: {url}")
                raise Exception("비디오 정보를 가져올 수 없습니다.")
            
            logger.info(f"[Task ID: {self.request.id}] YouTube 오디오 다운로드 중... ({url})")
            audio_path = youtube_processor.download_audio(url, output_dir=temp_dir)
            logger.info(f"[Task ID: {self.request.id}] 오디오 다운로드 완료: {audio_path}")

            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"[Task ID: {self.request.id}] 오디오 파일 경로를 얻지 못했거나 파일이 존재하지 않습니다.")
                raise FileNotFoundError("Downloaded audio file not found.")

            # 1. 타임스탬프 있는 트랜스크립션 수행
            logger.info(f"[Task ID: {self.request.id}] 타임스탬프 트랜스크립션 처리 시작: {audio_path}")
            transcript_segments = transcription_service.transcribe_with_timestamps(audio_path)
            
            # 2. 전체 텍스트 트랜스크립션
            logger.info(f"[Task ID: {self.request.id}] 전체 텍스트 STT 처리 시작: {audio_path}")
            full_text = transcription_service.transcribe_long_audio_from_file(audio_path)
            
            if not full_text and not transcript_segments:
                logger.warning(f"[Task ID: {self.request.id}] STT 결과가 비어있습니다. ({audio_path})")
                self.update_state(state='FAILURE', meta={'exc_type': 'EmptyResultError', 'exc_message': 'STT 결과가 비어있습니다.'})
                raise Exception("STT 결과가 비어있습니다.")

            logger.info(f"[Task ID: {self.request.id}] STT 처리 완료. 결과 길이: {len(full_text)} 문자, 세그먼트: {len(transcript_segments)}개")
            
            # 요약 기능
            summary = None
            if summarize and full_text:
                from services.summary_service import summarize_text_sync
                logger.info(f"[Task ID: {self.request.id}] 요약 작업 시작...")
                try:
                    summary = summarize_text_sync(full_text)
                    logger.info(f"[Task ID: {self.request.id}] 요약 작업 완료.")
                except Exception as e:
                    logger.error(f"[Task ID: {self.request.id}] 요약 중 오류 발생: {e}", exc_info=True)
            
            # 데이터베이스에 저장
            now = datetime.utcnow()
            document = {
                "video_id": video_id,
                "url": url,
                "title": video_info.get('title', ''),
                "transcript": transcript_segments,
                "full_text": full_text,
                "summary": summary,
                "created_at": now,
                "updated_at": now
            }
            
            transcript_collection.insert_one(document)
            logger.info(f"[Task ID: {self.request.id}] 트랜스크립트가 MongoDB에 저장되었습니다.")
            
            # 결과 반환
            result = {
                "transcript": full_text,
                "segments": transcript_segments,
                "summary": summary,
                "video_id": video_id,
                "title": video_info.get('title', ''),
                "url": url,
                "status": "new"
            }
            
            return result

        except FileNotFoundError as e_file:
            logger.error(f"[Task ID: {self.request.id}] 파일 관련 오류: {e_file}", exc_info=True)
            self.update_state(state='FAILURE', meta={'exc_type': type(e_file).__name__, 'exc_message': str(e_file)})
            raise
        except Exception as e:
            logger.error(f"[Task ID: {self.request.id}] YouTube 처리 작업 중 예기치 않은 오류 발생: {e}", exc_info=True)
            raise self.retry(exc=e, countdown=int(self.default_retry_delay * (self.request.retries + 1)))
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"[Task ID: {self.request.id}] 임시 오디오 파일 삭제 완료: {audio_path}")
                except OSError as e_remove:
                    logger.error(f"[Task ID: {self.request.id}] 임시 오디오 파일 삭제 실패: {e_remove}", exc_info=True)
            logger.info(f"[Task ID: {self.request.id}] YouTube 처리 작업 리소스 정리 완료.")