from openai import OpenAI

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

async def summarize(full_text: str) -> str:
    """GPT-4.1-mini 한글 요약 (5줄 이내 불릿)"""
    system_msg = (
        "당신은 한국어로된 강의 자막을 핵심 위주로 요약하는 AI 비서입니다. "
        "핵심 키워드를 보존하되 소제목과 불릿으로 정리하세요."
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": full_text},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
