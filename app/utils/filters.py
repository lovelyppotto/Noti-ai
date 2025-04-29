import re

IGNORE_WORDS = ["음", "어", "음..", "어..", "그.."]

def filter_text(text: str) -> str:
    """무의미한 단어와 배경음악 표현 필터링"""
    for word in IGNORE_WORDS:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    text = re.sub(r'[^\s]*(?:배경음악|음악)[^\s]*', '', text)
    text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
