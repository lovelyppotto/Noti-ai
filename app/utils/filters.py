import re
from typing import List, Pattern

# 필터링할 단어 목록 (추가 및 수정 가능)
# 의미 있는 약어 (AI, IT 등)는 제거되지 않도록 주의
DEFAULT_IGNORE_WORDS: List[str] = ["음", "어", "음...", "어...", "그...", "저...", "아..."]
# 짧은 영어 단어 중 예외적으로 허용할 단어 목록 (대소문자 구분 없음)
ALLOWED_SHORT_ENGLISH_WORDS: List[str] = ["AI", "IT", "ML", "DL", "UI", "UX", "API"]


class TextFilter:
    def __init__(self, ignore_words: List[str] = None, allowed_short_english: List[str] = None):
        """
        텍스트 필터링 클래스 초기화.

        Args:
            ignore_words (List[str], optional): 필터링할 추임새 목록. Defaults to DEFAULT_IGNORE_WORDS.
            allowed_short_english (List[str], optional): 허용할 짧은 영어 단어 목록. Defaults to ALLOWED_SHORT_ENGLISH_WORDS.
        """
        if ignore_words is None:
            ignore_words = DEFAULT_IGNORE_WORDS
        if allowed_short_english is None:
            allowed_short_english = ALLOWED_SHORT_ENGLISH_WORDS

        # 1. 무시할 단어 패턴 (단어 경계 확실히)
        # re.escape를 사용하여 정규식 특수 문자를 이스케이프 처리
        self.ignore_pattern: Pattern[str] = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, ignore_words)) + r')\b',
            re.IGNORECASE  # 대소문자 구분 없이
        )

        # 2. 배경음악/음악 관련 표현 제거 패턴
        # 좀 더 포괄적으로 수정 (예: "음악 소리", "배경 음악이 흐릅니다")
        self.music_pattern: Pattern[str] = re.compile(
            r'\b\S*(?:배경음악|음악)\S*\b|\b(?:음악이|배경음악이)\s.*?\b',
            re.IGNORECASE
        )

        # 3. 짧은 영어 단어 제거 패턴 (허용 목록 제외)
        # (?!)는 부정형 전방탐색(negative lookahead assertion)으로, 특정 패턴이 뒤따르지 않는 경우에만 매치
        # \b[a-zA-Z]{1,2}\b 패턴 앞에 허용 단어가 아닌 경우를 확인하는 조건 추가
        allowed_pattern_part = '|'.join(map(re.escape, allowed_short_english))
        self.short_english_pattern: Pattern[str] = re.compile(
            # 허용 목록에 없는 1~2자리 영어 단어만 제거
            r'\b(?!' + allowed_pattern_part + r'\b)([a-zA-Z]{1,2})\b',
            re.IGNORECASE # ALLOWED_SHORT_ENGLISH_WORDS도 대소문자 구분없이 비교하기 위함
        )
        
        # 4. 연속된 공백을 하나로 만드는 패턴
        self.multiple_spaces_pattern: Pattern[str] = re.compile(r'\s+')

    def filter_text(self, text: str) -> str:
        """
        주어진 텍스트에서 불필요한 단어, 표현, 공백을 필터링합니다.

        Args:
            text (str): 필터링할 원본 텍스트.

        Returns:
            str: 필터링된 텍스트.
        """
        if not text:
            return ""

        # 1. IGNORE_WORDS에 포함된 단어 제거
        text = self.ignore_pattern.sub('', text)

        # 2. 배경음악 관련 표현 제거
        text = self.music_pattern.sub('', text)

        # 3. 1~2자리의 영어 단어 제거 (허용 목록 제외)
        text = self.short_english_pattern.sub('', text)
        
        # 4. 연속된 공백을 하나로 만들고 양쪽 끝 공백 제거
        text = self.multiple_spaces_pattern.sub(' ', text).strip()
        
        # 필터링 후 문장 부호 주변의 어색한 공백 추가 정리 (예: " . " -> ". ")
        text = re.sub(r'\s+([.,?!])', r'\1', text)

        return text

# 기본 필터 인스턴스 (필요시 사용)
default_text_filter = TextFilter()