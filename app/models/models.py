from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# MongoDB를 위한 모델 정의
class TranscriptSegment(BaseModel):
    start: float = Field(..., description="시작 시간(초)")
    end: float = Field(..., description="종료 시간(초)")
    text: str = Field(..., description="해당 구간의 텍스트")

class VideoTranscript(BaseModel):
    video_id: str = Field(..., description="YouTube 비디오 ID")
    url: str = Field(..., description="YouTube 비디오 URL")
    title: Optional[str] = Field(None, description="비디오 제목")
    transcript: List[TranscriptSegment] = Field(default_factory=list, description="시간별 스크립트 세그먼트")
    full_text: str = Field("", description="전체 스크립트 텍스트")
    summary: Optional[str] = Field(None, description="요약 텍스트 (있는 경우)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="최종 업데이트 시간")
    
    class Config:
        schema_extra = {
            "example": {
                "video_id": "dQw4w9WgXcQ",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
                "transcript": [
                    {"start": 0.0, "end": 3.5, "text": "We're no strangers to love"},
                    {"start": 3.5, "end": 7.0, "text": "You know the rules and so do I"}
                ],
                "full_text": "We're no strangers to love\nYou know the rules and so do I",
                "summary": "클래식 팝 음악 비디오",
                "created_at": 1715155200,  # 2025-05-08의 Unix 타임스탬프
                "updated_at": 1715155200   # 2025-05-08의 Unix 타임스탬프
            }
        }