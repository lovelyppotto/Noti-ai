import logging
from openai import OpenAI, OpenAIError # OpenAIError로 구체적인 예외 처리
from config import settings

logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
# API 키는 환경 변수 OPENAI_API_KEY 또는 설정에서 가져옴
try:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
except Exception as e:
    logger.error(f"OpenAI 클라이언트 초기화 실패: {e}", exc_info=True)
    client = None # 클라이언트 초기화 실패 시 None으로 설정

async def summarize_text(full_text: str) -> str | None:
    if not client:
        logger.error("OpenAI 클라이언트가 초기화되지 않아 요약을 수행할 수 없습니다.")
        return None
        
    if not full_text or not full_text.strip():
        logger.warning("요약할 텍스트가 비어있습니다.")
        return "요약할 내용이 없습니다." # 또는 None 반환

    # 시스템 메시지 (요약 스타일 및 요구사항 정의)
    system_message_content = (
        "당신은 한국어로 작성된 긴 텍스트(주로 강의 또는 회의록 스크립트)를 핵심 내용 위주로 요약하는 AI 비서입니다. "
        "다음 지침을 따라 요약해주세요:\n"
        "1. 주요 내용을 명확하고 간결하게 전달해야 합니다.\n"
        "2. 핵심 키워드와 주제를 반드시 포함해야 합니다.\n"
        "3. 전체 내용을 포괄하는 소제목(## 제목 형식)을 먼저 제시하고, 그 아래에 불릿 포인트(-)를 사용하여 3~5가지 주요 항목으로 요약합니다.\n"
        "4. 각 불릿 포인트는 완전한 문장으로 작성하거나 핵심 구문으로 작성할 수 있습니다.\n"
        "5. 전체 요약은 한국어로 작성되어야 합니다."
    )
    
    user_message_content = f"다음 텍스트를 위의 지침에 따라 요약해주세요:\n\n---\n{full_text}\n---"

    try:
        logger.info(f"OpenAI 요약 요청 시작. 모델: {settings.OPENAI_SUMMARY_MODEL}, 텍스트 길이: {len(full_text)}")
        
        completion = client.chat.completions.create(
            model=settings.OPENAI_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content},
            ],
            max_tokens=settings.OPENAI_SUMMARY_MAX_TOKENS,
            temperature=settings.OPENAI_SUMMARY_TEMPERATURE,
            # top_p=1.0,
            # frequency_penalty=0.0,
            # presence_penalty=0.0,
        )
        
        summary = completion.choices[0].message.content.strip()
        logger.info(f"OpenAI 요약 성공. 요약된 텍스트 길이: {len(summary)}")
        return summary

    except OpenAIError as e: # OpenAI 라이브러리 관련 오류 처리
        logger.error(f"OpenAI API 오류 발생 (요약 중): {e}", exc_info=True)
        # e.http_status, e.code 등으로 더 자세한 오류 정보 확인 가능
        return f"요약 중 OpenAI API 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        logger.error(f"요약 처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
        return "요약 중 예기치 않은 오류가 발생했습니다."

# 동기 버전의 요약 함수 (Celery 등에서 사용하기 위함)
def summarize_text_sync(full_text: str) -> str | None:
    if not client:
        logger.error("OpenAI 클라이언트가 초기화되지 않아 요약을 수행할 수 없습니다.")
        return None
        
    if not full_text or not full_text.strip():
        logger.warning("요약할 텍스트가 비어있습니다.")
        return "요약할 내용이 없습니다."

    system_message_content = (
        "당신은 한국어로 작성된 긴 텍스트(주로 강의 또는 회의록 스크립트)를 핵심 내용 위주로 요약하는 AI 비서입니다. "
        "다음 지침을 따라 요약해주세요:\n"
        "1. 주요 내용을 명확하고 간결하게 전달해야 합니다.\n"
        "2. 핵심 키워드와 주제를 반드시 포함해야 합니다.\n"
        "3. 전체 내용을 포괄하는 소제목(## 제목 형식)을 먼저 제시하고, 그 아래에 불릿 포인트(-)를 사용하여 3~5가지 주요 항목으로 요약합니다.\n"
        "4. 각 불릿 포인트는 완전한 문장으로 작성하거나 핵심 구문으로 작성할 수 있습니다.\n"
        "5. 전체 요약은 한국어로 작성되어야 합니다."
    )
    user_message_content = f"다음 텍스트를 위의 지침에 따라 요약해주세요:\n\n---\n{full_text}\n---"

    try:
        logger.info(f"OpenAI 동기 요약 요청 시작. 모델: {settings.OPENAI_SUMMARY_MODEL}, 텍스트 길이: {len(full_text)}")
        completion = client.chat.completions.create(
            model=settings.OPENAI_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content},
            ],
            max_tokens=settings.OPENAI_SUMMARY_MAX_TOKENS,
            temperature=settings.OPENAI_SUMMARY_TEMPERATURE,
        )
        summary = completion.choices[0].message.content.strip()
        logger.info(f"OpenAI 동기 요약 성공. 요약된 텍스트 길이: {len(summary)}")
        return summary
    except OpenAIError as e:
        logger.error(f"OpenAI API 오류 발생 (동기 요약 중): {e}", exc_info=True)
        return f"요약 중 OpenAI API 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        logger.error(f"동기 요약 처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
        return "요약 중 예기치 않은 오류가 발생했습니다."