import torch
import numpy as np
import logging
from functools import lru_cache
from config import settings # 설정 가져오기


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_vad_model_and_utils():
    """
    Silero VAD 모델과 유틸리티 함수를 로드하고 캐시합니다.
    모델 로드 실패 시 None을 반환합니다.
    """
    try:
        logger.info(f"VAD 모델 로드 시도: {settings.VAD_MODEL_NAME} from {settings.VAD_MODEL_REPO_OR_DIR}")
        vad_model, utils = torch.hub.load(
            repo_or_dir=settings.VAD_MODEL_REPO_OR_DIR,
            model=settings.VAD_MODEL_NAME,
            force_reload=settings.VAD_FORCE_RELOAD, # 개발 중에는 True, 프로덕션에서는 False 권장
            trust_repo=True # trust_repo=True 추가 (최신 torch.hub 요구사항)
        )
        logger.info("VAD 모델이 성공적으로 로드되었습니다.")
        return vad_model, utils
    except Exception as e:
        logger.error(f"VAD 모델 로드 실패: {e}", exc_info=True)
        return None, None

def get_vad_device():
    """VAD 모델을 실행할 장치를 결정합니다."""
    if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
        logger.info("VAD에 CUDA 장치를 사용합니다.")
        return "cuda"
    logger.info("VAD에 CPU 장치를 사용합니다.")
    return "cpu"

# VAD 모델 및 유틸리티 가져오기
_vad_model, _vad_utils = get_vad_model_and_utils()
_vad_device = get_vad_device()

if _vad_model and _vad_utils:
    _get_speech_timestamps = _vad_utils[0] # get_speech_timestamps 함수
    _vad_model = _vad_model.to(_vad_device) # 모델을 적절한 장치로 이동
else:
    logger.warning("VAD 모델을 사용할 수 없습니다. is_speech 함수는 항상 False를 반환합니다.")
    _get_speech_timestamps = None # 모델 로드 실패 시 None으로 설정


def decode_audio_from_bytes(binary_audio: bytes, sample_width: int = 2, channels: int = 1) -> np.ndarray:
    if not binary_audio:
        return np.array([], dtype=np.float32)
    try:
        # 바이트 데이터를 적절한 NumPy dtype으로 변환
        if sample_width == 2: # 16-bit
            dtype = np.int16
        elif sample_width == 1: # 8-bit
            dtype = np.uint8 # 또는 np.int8, 오디오 소스에 따라 다름
        elif sample_width == 4: # 32-bit int or float
             # 일반적으로 float32로 직접 오는 경우는 드물지만, int32일 수 있음
            dtype = np.int32
        else:
            logger.error(f"지원되지 않는 sample_width: {sample_width}")
            return np.array([], dtype=np.float32)

        audio_np = np.frombuffer(binary_audio, dtype=dtype)

        # 스테레오인 경우 모노로 변환 (첫 번째 채널 사용 또는 평균)
        if channels > 1:
            audio_np = audio_np[::channels] # 간단히 첫 번째 채널만 사용

        # float32로 정규화 (-1.0 ~ 1.0 범위)
        if dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif dtype == np.uint8: # 8-bit unsigned
            audio_np = (audio_np.astype(np.float32) - 128.0) / 128.0
        elif dtype == np.int8: # 8-bit signed
            audio_np = audio_np.astype(np.float32) / 128.0
        elif dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        # 이미 float32인 경우는 정규화 가정

        return audio_np.clip(-1.0, 1.0) # 안전장치

    except Exception as e:
        logger.error(f"오디오 디코딩 중 오류 발생: {e}", exc_info=True)
        return np.array([], dtype=np.float32)


def is_speech_present(audio_np: np.ndarray, sample_rate: int = settings.WHISPER_SAMPLING_RATE) -> bool:
    """
    VAD를 사용하여 주어진 오디오 데이터(NumPy 배열)에 음성이 있는지 확인합니다.

    Args:
        audio_np (np.ndarray): Float32 PCM 오디오 데이터.
        sample_rate (int): 오디오 샘플링 레이트.

    Returns:
        bool: 음성이 감지되면 True, 아니면 False. VAD 모델 로드 실패 시 항상 False.
    """
    if not _get_speech_timestamps or not _vad_model:
        logger.debug("VAD 모델이 로드되지 않아 음성 감지를 건너<0xEB><0><0x8F><0xBB>니다.")
        # VAD 모델이 없으면, 모든 오디오를 음성으로 간주하거나 (True 반환)
        # 또는 음성 없음으로 간주 (False 반환) 할 수 있습니다.
        # 여기서는 False를 반환하여 VAD가 필수적이지 않은 경우 STT가 모든 오디오를 처리하도록 유도할 수 있습니다.
        # 혹은, True를 반환하여 STT가 항상 시도하도록 할 수 있습니다.
        # 현재 로직에서는 whisper_service에서 is_speech 결과가 False면 STT를 안하므로,
        # VAD 실패 시 STT가 동작하지 않게 됩니다. 이를 원치 않으면 True를 반환하도록 수정.
        # 여기서는 VAD 실패 시에도 STT를 시도하도록 True를 반환하는 것이 더 안전할 수 있습니다.
        # return True # VAD 실패 시 STT 시도
        return False # VAD 실패 시 STT 시도 안함 (기존 로직 유지)

    if audio_np.size == 0:
        return False

    try:
        audio_tensor = torch.from_numpy(audio_np).to(_vad_device)
        # Silero VAD는 1차원 텐서를 기대합니다.
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0) # 스테레오면 모노로

        speech_timestamps = _get_speech_timestamps(
            audio_tensor,
            _vad_model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=settings.VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=settings.VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=settings.VAD_SPEECH_PAD_MS
        )
        # logger.debug(f"VAD Speech timestamps: {speech_timestamps}")
        return len(speech_timestamps) > 0
    except Exception as e:
        logger.error(f"VAD 음성 감지 중 오류 발생: {e}", exc_info=True)
        return False # 오류 발생 시 음성 없음으로 처리