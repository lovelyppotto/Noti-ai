from pymongo import MongoClient
from config import settings

# MongoDB 연결 설정
client = MongoClient(settings.MONGODB_URI)
db = client[settings.MONGODB_DB_NAME]
transcript_collection = db[settings.MONGODB_TRANSCRIPT_COLLECTION]

# 컬렉션에 필요한 인덱스 생성
transcript_collection.create_index("video_id", unique=True)