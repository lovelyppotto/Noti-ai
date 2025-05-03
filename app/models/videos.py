from pydantic import BaseModel, HttpUrl

class VideoReq(BaseModel):
    url: HttpUrl
    summarize: bool = False
