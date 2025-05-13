import logging
import numpy as np
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException, status

from services.vad_service import decode_audio_from_bytes 
from services.whisper_service import TranscriptionService
from services.summary_service import summarize_text 
from config import settings


logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/realtime-stt") # 경로 명확히 변경
async def websocket_stt_endpoint(ws: WebSocket):
    """
    실시간 STT WebSocket 엔드포인트.
    클라이언트로부터 오디오 바이트를 받아 STT 결과를 실시간으로 전송하고,
    요청 시 전체 STT 기록에 대한 요약을 제공합니다.
    """
    await ws.accept()
    logger.info(f"클라이언트 연결됨: {ws.client}")

    # 각 연결에 대한 독립적인 상태 변수 초기화
    transcription_service = TranscriptionService() # STT 서비스 인스턴스 생성
    if not all([transcription_service.model, transcription_service.processor, transcription_service.tokenizer]):
        logger.error(f"STT 서비스 초기화 실패 (클라이언트: {ws.client}). 연결을 종료합니다.")
        await ws.close(code=status.WS_1011_INTERNAL_ERROR, reason="STT service initialization failed")
        return

    # 오디오 버퍼 (NumPy 배열 저장)
    audio_buffer_np = deque()
    # Whisper 프롬프트용 토큰 히스토리 (토큰 ID 저장)
    token_hist = deque(maxlen=settings.WHISPER_TOKEN_HISTORY_MAXLEN)
    # 요약 및 컨텍스트용 전체 텍스트 히스토리
    current_full_history_txt = ""
    
    MIN_SAMPLES_FOR_STT = int(settings.WHISPER_SAMPLING_RATE * 1.0) # 1초 분량의 오디오를 모아서 처리

    try:
        while True:
            #
            
            received_data = await ws.receive() # receive_json(), receive_bytes(), receive_text()

            if "bytes" in received_data: # 오디오 데이터 처리 (클라이언트가 {'bytes': <audio_data_bytes>} 형태로 보낸다고 가정)
                audio_bytes = received_data["bytes"]
                if not isinstance(audio_bytes, bytes):
                    logger.warning(f"잘못된 오디오 데이터 타입 수신: {type(audio_bytes)}. 무시합니다.")
                    continue

                audio_np_segment = decode_audio_from_bytes(audio_bytes, sample_width=2, channels=1)

                if audio_np_segment.size > 0:
                    audio_buffer_np.append(audio_np_segment)
                    
                    # 버퍼에 쌓인 총 샘플 수 계산
                    total_samples_in_buffer = sum(chunk.shape[0] for chunk in audio_buffer_np)

                    # 충분한 오디오 데이터가 모이면 STT 처리
                    if total_samples_in_buffer >= MIN_SAMPLES_FOR_STT:
                        # 버퍼의 모든 오디오 청크를 하나로 합침
                        combined_audio_np = np.concatenate(list(audio_buffer_np))
                        audio_buffer_np.clear() # 버퍼 비우기

                        logger.debug(f"STT 처리할 오디오 길이: {combined_audio_np.shape[0]} 샘플")
                        
                        # STT 실행
                        # transcribe_chunk는 (filtered_text, updated_history_txt, updated_token_hist) 반환
                        transcribed_text, updated_hist_txt, updated_token_hist = \
                            transcription_service.transcribe_chunk(combined_audio_np, token_hist, current_full_history_txt)
                        
                        # 상태 업데이트
                        current_full_history_txt = updated_hist_txt
                        token_hist = updated_token_hist # deque는 가변 객체이므로 재할당 또는 내부 수정 모두 가능

                        if transcribed_text:
                            logger.info(f"STT 결과 (클라이언트: {ws.client}): {transcribed_text}")
                            await ws.send_text(transcribed_text) # 클라이언트에 STT 결과 전송
                        else:
                            logger.debug(f"STT 결과 없음 (클라이언트: {ws.client})")

            elif "text" in received_data: # 텍스트 명령어 처리 (클라이언트가 {'text': '__SUMMARY__'} 형태로 보낸다고 가정)
                command = received_data["text"]
                if command == "__SUMMARY__":
                    logger.info(f"요약 요청 수신 (클라이언트: {ws.client}). 히스토리 길이: {len(current_full_history_txt)}")
                    if not current_full_history_txt.strip():
                        summary_response = "[요약할 내용이 없습니다]"
                        logger.info("요약할 텍스트 히스토리가 비어있습니다.")
                    else:
                        try:
                            summary = await summarize_text(current_full_history_txt) # 비동기 요약 함수 호출
                            summary_response = f"[요약]\n{summary}"
                            logger.info(f"요약 생성 완료 (클라이언트: {ws.client}):\n{summary}")
                        except Exception as e:
                            logger.error(f"요약 생성 중 오류 발생 (클라이언트: {ws.client}): {e}", exc_info=True)
                            summary_response = "[요약 생성 중 오류가 발생했습니다]"
                    
                    await ws.send_text(summary_response) # 클라이언트에 요약 결과 전송
                else:
                    logger.warning(f"알 수 없는 텍스트 명령어 수신: {command} (클라이언트: {ws.client})")
            else:
                # 예상치 못한 데이터 형식
                logger.warning(f"예상치 못한 데이터 형식 수신 (클라이언트: {ws.client}): {type(received_data)}. 무시합니다.")


    except WebSocketDisconnect:
        logger.info(f"클라이언트 연결 끊김: {ws.client}")
    except WebSocketException as e: # FastAPI WebSocketException 처리
        logger.error(f"WebSocket 오류 발생 (클라이언트: {ws.client}): {e.reason} (code: {e.code})", exc_info=True)
        # await ws.close(code=e.code, reason=e.reason) # 이미 FastAPI가 처리할 수 있음
    except Exception as e:
        logger.error(f"WebSocket 처리 중 예기치 않은 오류 발생 (클라이언트: {ws.client}): {e}", exc_info=True)
        # 연결이 아직 살아있다면 오류 메시지와 함께 연결 종료 시도
        if ws.client_state == ws.client_state.CONNECTED: # type: ignore
            await ws.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal server error")
    finally:
        # 연결 종료 시 리소스 정리 (필요한 경우)
        logger.info(f"WebSocket 연결 정리 (클라이언트: {ws.client})")
        # 예: transcription_service 인스턴스 관련 정리 (특별히 필요한 것이 없다면 생략)
        del transcription_service
        del audio_buffer_np
        del token_hist
        del current_full_history_txt

