import os
from pathlib import Path

class Settings:
    # 기본 설정
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # API 키
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = 'key'    
    
    # 챗봇 설정
    CHAT_MODEL = "gpt-4-mini"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    
    # 벡터 검색 설정
    VECTOR_DIMENSION = 1536
    SIMILAR_CONVERSATIONS_COUNT = 5
    
    # interaction threshold 설정
    DEFAULT_INTERACTION_THRESHOLD = 5
    
    # 데이터베이스 설정
    DATABASE_HOST = ''
    DATABASE_URL = 'jdbc:mysql://host:3306/bf?serverTimezone=Asia/Seoul'
    DATABASE_USER = 'root'
    DATABASE_PASSWORD = '1234'
    DATABASE_NAME = 'bf'