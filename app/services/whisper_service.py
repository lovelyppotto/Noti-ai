import torch
import numpy as np
from collections import deque
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperTokenizerFast
from functools import lru_cache
import logging
import os # 임시 파일 삭제용
import librosa
import re

# 설정 및 필터 가져오기
from config import settings
from utils.filters import TextFilter

logger = logging.getLogger(__name__)
default_text_filter = TextFilter(
    ignore_words=settings.FILTER_IGNORE_WORDS,
    allowed_short_english=settings.FILTER_ALLOWED_SHORT_ENGLISH_WORDS
)


def get_whisper_device():
    """Whisper 모델을 실행할 장치를 결정합니다."""
    if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
        logger.info("Whisper에 CUDA 장치를 사용합니다.")
        return "cuda"
    logger.info("Whisper에 CPU 장치를 사용합니다.")
    return "cpu"


_whisper_device = get_whisper_device()

# _torch_dtype 설정 부분 수정
try:
    _torch_dtype = getattr(torch, settings.WHISPER_TORCH_DTYPE, torch.float16)
    # 호환성 검사
    if _whisper_device == "cuda" and _torch_dtype not in [torch.float16, torch.bfloat16]:
        logger.warning(f"CUDA에서 {_torch_dtype}는 권장되지 않음, float16으로 대체")
        _torch_dtype = torch.float16
    elif _whisper_device == "cpu" and _torch_dtype != torch.float32:
        logger.warning(f"CPU에서 {_torch_dtype}는 권장되지 않음, float32로 대체")
        _torch_dtype = torch.float32
except AttributeError as e:
    logger.warning(f"torch dtype {settings.WHISPER_TORCH_DTYPE} 설정 실패: {e}, 기본값 사용")
    _torch_dtype = torch.float16 if _whisper_device == "cuda" else torch.float32


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """librosa를 사용한 오디오 로드 함수"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        logger.error(f"오디오 파일 로드 중 오류: {e}")
        return np.array([])


@lru_cache(maxsize=1)
def get_whisper_model_instance():
    """
    Whisper 모델 및 프로세서/토크나이저를 로드하고 캐시합니다.
    실패 시 None을 반환합니다.
    """
    try:
        logger.info(f"Whisper 모델 로드 시도: {settings.WHISPER_MODEL_NAME}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            settings.WHISPER_MODEL_NAME,
            torch_dtype=_torch_dtype,
            low_cpu_mem_usage=settings.WHISPER_LOW_CPU_MEM_USAGE,
            use_safetensors=settings.WHISPER_USE_SAFETENSORS
        ).to(_whisper_device)
        model.eval() # 추론 모드로 설정

        processor = AutoProcessor.from_pretrained(settings.WHISPER_MODEL_NAME)
        # WhisperTokenizerFast 사용 명시 (get_prompt 내부 로직과 호환성)
        tokenizer = WhisperTokenizerFast.from_pretrained(settings.WHISPER_MODEL_NAME)

        logger.info(f"Whisper 모델 ({settings.WHISPER_MODEL_NAME}) 및 프로세서/토크나이저가 {_whisper_device}에 성공적으로 로드되었습니다.")
        return model, processor, tokenizer
    except Exception as e:
        logger.error(f"Whisper 모델 또는 프로세서 로드 실패: {e}", exc_info=True)
        return None, None, None


# VAD 서비스에서 음성 감지 함수 가져오기
from services.vad_service import is_speech_present


class TranscriptionService:
    def __init__(self, text_filter: TextFilter = default_text_filter):
        """
        STT 서비스를 초기화합니다.

        Args:
            text_filter (TextFilter, optional): 사용할 텍스트 필터. Defaults to default_text_filter.
        """
        self.model, self.processor, self.tokenizer = get_whisper_model_instance()
        self.text_filter = text_filter

        if not all([self.model, self.processor, self.tokenizer]):
            logger.error("TranscriptionService 초기화 실패: 모델/프로세서/토크나이저 로드 실패.")
            # 필요시 여기서 예외 발생 또는 플래그 설정

    def _get_prompt(self, token_hist: deque) -> str:
        """이전 토큰 히스토리를 기반으로 프롬프트를 생성합니다."""
        base_prompt = " 다음은 이전 대화 내용입니다. 이어서 자연스럽게 변환해주세요. 영어 표현은 영어로, 한글은 한글로 정확히 변환해주세요. 배경 소음이나 음악은 무시하세요."
        try:
            history_prompt_text = self.tokenizer.decode(list(token_hist), skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"토큰 히스토리 디코딩 중 오류: {e}. 빈 히스토리로 처리합니다.")
            history_prompt_text = ""
        return base_prompt + history_prompt_text

    def transcribe_chunk(self, audio_np: np.ndarray, token_hist: deque, current_history_txt: str) -> tuple[str, str, deque]:
        """
        단일 오디오 청크를 STT 처리합니다.
        """
        if not all([self.model, self.processor, self.tokenizer]):
            logger.warning("모델이 로드되지 않아 STT를 건너뜁니다.")
            return "", current_history_txt, token_hist

        if not is_speech_present(audio_np, sample_rate=settings.WHISPER_SAMPLING_RATE):
            logger.debug("VAD 결과 음성 미감지, STT 건너뜁니다.")
            return "", current_history_txt, token_hist

        try:
            # prompt_text = self._get_prompt(token_hist) # 프롬프트 사용은 모델 및 라이브러리 버전에 따라 조정 필요
            
            inputs = self.processor(
                audio_np,
                sampling_rate=settings.WHISPER_SAMPLING_RATE,
                return_tensors="pt",
            )
            
            input_features = inputs.input_features.to(_whisper_device)
            if input_features.dtype != _torch_dtype and _whisper_device == "cuda":
                input_features = input_features.to(_torch_dtype)

            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features,
                    language="ko",  # 한국어 명시
                    task="transcribe",  # 번역이 아닌 트랜스크립션
                    max_new_tokens=128
                )
            
            transcribed_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.debug(f"STT 원본 결과: {transcribed_text}")

            filtered_text = self.text_filter.filter_text(transcribed_text)
            logger.debug(f"STT 필터링 결과: {filtered_text}")

            if not filtered_text:
                return "", current_history_txt, token_hist

            new_tokens = self.tokenizer.encode(filtered_text, add_special_tokens=False)
            token_hist.extend(new_tokens)

            updated_history_txt = (current_history_txt + filtered_text + " ").strip()
            if len(updated_history_txt) > settings.WHISPER_MAX_CHARS_HISTORY:
                updated_history_txt = updated_history_txt[-settings.WHISPER_MAX_CHARS_HISTORY:]
            
            return filtered_text.strip(), updated_history_txt, token_hist

        except Exception as e:
            logger.error(f"STT 처리 중 오류 발생: {e}", exc_info=True)
            return "", current_history_txt, token_hist

    def transcribe_long_audio_from_file(self, file_path: str) -> str:
        """장시간 오디오 파일 처리 (YouTube용 등)"""
        if not all([self.model, self.processor, self.tokenizer]):
            logger.warning("모델이 로드되지 않아 긴 오디오 STT를 건너뜁니다.")
            return ""
        
        if not os.path.exists(file_path):
            logger.error(f"오디오 파일을 찾을 수 없습니다: {file_path}")
            return ""
    
        try:
            logger.info(f"긴 오디오 파일 STT 시작: {file_path}")
            
            # 오디오 로드
            audio = load_audio(file_path, sr=settings.WHISPER_SAMPLING_RATE)
            if len(audio) == 0:
                logger.error(f"오디오 파일 로드 실패: {file_path}")
                return ""
            
            # 청크 단위로 처리
            chunk_size = settings.WHISPER_SAMPLING_RATE * settings.WHISPER_CHUNK_SIZE_SECONDS
            num_chunks = (len(audio) + chunk_size - 1) // chunk_size
            
            logger.info(f"오디오를 {num_chunks}개의 청크로 나누어 처리합니다.")
            
            transcripts = []
            
            for i in range(0, len(audio), chunk_size):
                audio_chunk = audio[i:i + chunk_size]
                
                if len(audio_chunk) < settings.WHISPER_SAMPLING_RATE:  # 1초 미만 청크는 무시
                    continue
                    
                # 프로세서로 처리
                inputs = self.processor(
                    audio_chunk, 
                    sampling_rate=settings.WHISPER_SAMPLING_RATE,
                    return_tensors="pt"
                )
                
                # 데이터 타입 일치
                input_features = inputs.input_features.to(_whisper_device)
                if input_features.dtype != _torch_dtype:
                    input_features = input_features.to(_torch_dtype)
                
                # 트랜스크립션 수행
                with torch.inference_mode():
                    result = self.model.generate(
                        input_features,
                        language="ko",
                        task="transcribe",
                        max_new_tokens=256
                    )
                
                # 결과 디코딩
                chunk_text = self.processor.batch_decode(result, skip_special_tokens=True)[0]
                filtered_text = self.text_filter.filter_text(chunk_text)
                
                if filtered_text.strip():
                    transcripts.append(filtered_text)
                    logger.debug(f"청크 {i//chunk_size + 1}/{num_chunks} STT: {filtered_text[:100]}...")
            
            full_transcript = "\n".join(transcripts).strip()
            logger.info(f"긴 오디오 파일 STT 완료. 총 길이: {len(full_transcript)} 문자")
            return full_transcript
    
        except Exception as e:
            logger.error(f"긴 오디오 STT 처리 중 오류 발생 ({file_path}): {e}", exc_info=True)
            return ""

    def transcribe_with_timestamps(self, audio_path: str) -> list:
        """오디오 파일을 트랜스크립션하고 타임스탬프를 포함한 결과 반환"""
        if not all([self.model, self.processor, self.tokenizer]):
            logger.warning("모델이 로드되지 않아 타임스탬프 트랜스크립션을 건너뜁니다.")
            return []
        
        try:
            logger.info(f"타임스탬프 트랜스크립션 시작: {audio_path}")
            
            # 오디오 로드
            audio = load_audio(audio_path, sr=settings.WHISPER_SAMPLING_RATE)
            if len(audio) == 0:
                logger.error(f"오디오 파일 로드 실패: {audio_path}")
                return []
            
            # 간단한 세그먼트 생성
            segments = []
            chunk_size = settings.WHISPER_SAMPLING_RATE * 30  # 30초 단위로 처리
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size / 10:  # 너무 짧은 청크는 무시
                    continue
                    
                # 패딩 처리
                chunk_inputs = self.processor(
                    chunk, 
                    sampling_rate=settings.WHISPER_SAMPLING_RATE,
                    return_tensors="pt"
                )
                chunk_features = chunk_inputs.input_features.to(_whisper_device)
                if chunk_features.dtype != _torch_dtype:
                    chunk_features = chunk_features.to(_torch_dtype)
                
                # 트랜스크립션 수행
                with torch.inference_mode():
                    # 언어 명시적 지정
                    try:
                        result = self.model.generate(
                            chunk_features,
                            language="ko",  # 한국어로 명시적 지정
                            task="transcribe",  # 번역 아닌 트랜스크립션
                            return_timestamps=True
                        )
                    except Exception as e:
                        logger.warning(f"타임스탬프 생성 시도 실패: {e}")
                        # 단순 트랜스크립션 시도
                        result = self.model.generate(
                            chunk_features,
                            language="ko",
                            task="transcribe",
                            max_new_tokens=256
                        )
                
                # 특수 토큰 제거한 텍스트 추출
                text = self.processor.batch_decode(result, skip_special_tokens=True)[0]
                text = self.text_filter.filter_text(text)
                
                if text.strip():  # 빈 텍스트가 아닐 경우만 추가
                    # 타임스탬프 계산
                    start_time = i / settings.WHISPER_SAMPLING_RATE
                    end_time = min((i + len(chunk)) / settings.WHISPER_SAMPLING_RATE, 
                                   len(audio) / settings.WHISPER_SAMPLING_RATE)
                    
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
            
            if not segments:
                logger.warning("타임스탬프 세그먼트 생성 실패, 대체 방법 시도...")
                return self._create_simple_timestamps(audio_path)
            
            return segments
            
        except Exception as e:
            logger.error(f"타임스탬프 트랜스크립션 중 오류 발생: {e}", exc_info=True)
            logger.warning("타임스탬프 오류, 간단한 대체 방법 시도...")
            try:
                return self._create_simple_timestamps(audio_path)
            except Exception as fallback_e:
                logger.error(f"대체 타임스탬프 방법도 실패: {fallback_e}", exc_info=True)
                return []

    def _create_simple_timestamps(self, audio_path: str) -> list:
        """전체 텍스트에서 간단한 타임스탬프 생성 (백업 방법)"""
        try:
            # 전체 텍스트 가져오기
            full_text = self.transcribe_long_audio_from_file(audio_path)
            
            if not full_text:
                return []
            
            # 오디오 길이 파악
            audio, sr = librosa.load(audio_path, sr=None)
            audio_length_sec = len(audio) / sr
            
            # 텍스트 분할 (간단하게 문장 단위)
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            sentences = [s for s in sentences if s.strip()]
            
            # 오디오 길이에 비례하여 타임스탬프 분배
            segments = []
            total_chars = sum(len(s) for s in sentences)
            
            current_time = 0.0
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # 문장 길이에 비례하여 시간 할당
                duration = (len(sentence) / total_chars) * audio_length_sec if total_chars > 0 else 5.0
                
                segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": sentence.strip()
                })
                
                current_time += duration
                
            return segments
        
        except Exception as e:
            logger.error(f"간단한 타임스탬프 생성 중 오류: {e}", exc_info=True)
            return []