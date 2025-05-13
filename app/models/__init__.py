from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

# MongoDB 연결 설정
client = MongoClient(settings.MONGODB_URI)
db = client[settings.MONGODB_DB_NAME]
transcript_collection = db[settings.MONGODB_TRANSCRIPT_COLLECTION]

# 비동기 클라이언트 (FastAPI 엔드포인트용)
async_mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
async_db = async_mongo_client[settings.MONGODB_DB_NAME]
async_transcript_collection = async_db[settings.MONGODB_TRANSCRIPT_COLLECTION]

# 컬렉션에 필요한 인덱스 생성
transcript_collection.create_index("video_id", unique=True)