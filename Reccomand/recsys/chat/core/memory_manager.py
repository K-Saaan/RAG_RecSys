import logging
import asyncio
import aiomysql
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import faiss
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import mysql.connector
from mysql.connector import Error

from config.settings import Settings
from core.models import REC_LOG
from utils.validators import InputValidator

# 로깅 설정
logging.basicConfig(
    filename="log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class ChatbotMemoryManager:
    def __init__(self, settings):
        self.settings = settings
        logging.info("ChatbotMemoryManager 초기화 시작")
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )

        self.pool = None
        self.index = faiss.IndexFlatL2(settings.VECTOR_DIMENSION)
        self.user_metadata = []
        self.exercise_metadata = []
        self.interaction_metadata = []

        self.user_vector = faiss.IndexFlatL2(settings.VECTOR_DIMENSION)
        self.exercise_vector = faiss.IndexFlatL2(settings.VECTOR_DIMENSION)
        self.interaction_vector = faiss.IndexFlatL2(settings.VECTOR_DIMENSION)

        self.conversations: List[REC_LOG] = []
        self.last_interaction: Dict[str, datetime] = {}

        self.db_connection = self.get_mysql_connection()
        logging.info("ChatbotMemoryManager 초기화 완료")

    def get_mysql_connection(self):
        try:
            connection = mysql.connector.connect(
                host=Settings.DATABASE_HOST,
                user=Settings.DATABASE_USER,
                password=Settings.DATABASE_PASSWORD,
                database=Settings.DATABASE_NAME
            )
            if connection.is_connected():
                print("MySQL 데이터베이스에 연결되었습니다.")
            return connection
        except Error as e:
            print(f"MySQL 연결 오류: {e}")
            return None

    async def create_connection_pool(self):
        try:
            logging.info("MySQL 연결 풀 생성 시작")
            self.pool = await aiomysql.create_pool(
                host=self.settings.DATABASE_HOST,
                user=self.settings.DATABASE_USER,
                password=self.settings.DATABASE_PASSWORD,
                db=self.settings.DATABASE_NAME,
                autocommit=True
            )
            logging.info("MySQL 연결 풀 생성 완료")
        except Error as e:
            logging.error(f"MySQL 연결 오류: {e}")
            return None
    
    # 쿼리 실행
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        try:
            if self.pool is None:
                raise ValueError("MySQL 연결 풀이 초기화되지 않았습니다.")
            logging.info(f"Query : {query}")
            logging.info(f"Params : {params}")
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params or ())
                    return await cursor.fetchall()
        except Exception as e:
            logging.error(f"MySQL Query 오류: {e}")

    async def preload_data(self):
        """데이터 로드 및 FAISS 벡터 저장소 초기화"""
        # 사용자 데이터 로드
        logging.info("사용자 데이터 로드")
        user_ids = [23, 24, 25, 26, 27, 28]
        placeholders = ", ".join(["%s"] * len(user_ids)) 
        logging.info(f"user_ids : {user_ids}")
        logging.info(f"placeholders : {placeholders}")
        u_query = f"""
                SELECT user_index, age, gender, disability, disability_detail, disability_rank, 
                       exercise_experience, exercise_purpose
                FROM users
                
            """
        # WHERE user_index IN ({placeholders})
        # users = await self.execute_query(u_query ,tuple(user_ids))
        users = await self.execute_query(u_query)
        user_meta = [
            f"{user['user_index']} {user['age']} {user['gender']} {user['disability']} {user['exercise_purpose']}"
            for user in users
        ]
        logging.info(f"조회된 사용자 수 {len(users)}")

        # FAISS 벡터 저장소 생성
        if user_meta:
            logging.info("사용자 데이터를 벡터화하고 user_vector 인덱스를 초기화합니다.")
            user_vectors = [await self.embeddings.aembed_query(text) for text in user_meta]
            user_vectors = np.array(user_vectors, dtype=np.float32)
            
            # 벡터 추가
            self.user_vector.add(user_vectors)
            for i in range(len(user_meta)):           
                self.user_metadata.append(users[i])
            logging.info(f"user_vector 인덱스에 {len(user_vectors)}개의 사용자 데이터를 추가했습니다.")

        # 운동 데이터 로드
        logging.info("운동 데이터 로드")
        ex_query = f"""
            SELECT ftness_mesure_index, age_flag_nm, mesure_age_co, sexdstn_flag_cd, 
                    trobl_ty_nm, trobl_detail_nm, trobl_grad_nm, exercise_stage, exercise_name
            FROM ftness_mesure_data
            LIMIT 50
            """
        exercises = await self.execute_query(ex_query)
        
        # self.user_cache = {user['user_index']: user for user in users}
        exercise_meta = [
            f"{exercise['exercise_name']} {exercise['age_flag_nm']} {exercise['trobl_ty_nm']} {exercise['trobl_grad_nm']}"
            for exercise in exercises
        ]
        logging.info(f"조회된 운동 수 {len(exercises)}")
        
        # FAISS 벡터 저장소 생성
        if exercise_meta:
            logging.info("운동 데이터를 벡터화하고 exercise_vectors 인덱스를 초기화합니다.")
            exercise_vectors = [await self.embeddings.aembed_query(text) for text in exercise_meta]
            exercise_vectors = np.array(exercise_vectors, dtype=np.float32)
            
            # 벡터 추가
            self.exercise_vector.add(exercise_vectors)
            logging.info(f"exercise_meta 수 : {len(exercise_meta)}")
            for i in range(len(exercise_meta)):
                self.exercise_metadata.append(exercises[i])
            logging.info(f"exercise_vectors 인덱스에 {len(exercise_vectors)}개의 운동 데이터를 추가했습니다.")
            logging.info(f"exercise_metadata 인덱스에 {len(self.exercise_metadata)}개의 운동 데이터를 추가했습니다.")

        # 운동 선호 데이터 로드
        logging.info("운동 선호 데이터 로드")
        user_ids = [23, 24, 25, 26, 27, 28]
        placeholders = ", ".join(["%s"] * len(user_ids)) 
        if_query = f"""
            SELECT reco_ftness_index, user_index, exercise_name, like_yn
            FROM reco_ftness
            
            """
        # WHERE user_index IN ({placeholders})
        # interactions = await self.execute_query(if_query, tuple(user_ids))
        interactions = await self.execute_query(if_query)
        interation_meta = [
            f"{interaction['reco_ftness_index']} {interaction['user_index']} {interaction['exercise_name']} {interaction['like_yn']}"
            for interaction in interactions
        ]
        logging.info(f"조회된 운동 수 {len(interactions)}")
        # FAISS 벡터 저장소 생성
        if interation_meta:
            logging.info("운동 선호 데이터를 벡터화하고 exercise_vectors 인덱스를 초기화합니다.")
            interaction_vectors = [await self.embeddings.aembed_query(text) for text in interation_meta]
            interaction_vectors = np.array(interaction_vectors, dtype=np.float32)
            
            # 벡터 추가
            self.interaction_vector.add(interaction_vectors)
            for i in range(len(interation_meta)):
                self.interaction_metadata.append(interactions[i]) 
            logging.info(f"interaction_vectors 인덱스에 {len(interaction_vectors)}개의 운동 선호 데이터를 추가했습니다.")
    
    # 특정 사용자의 상호작용 횟수를 조회
    async def get_user_interaction_count(self, user_id: int) -> int:
        try:
            logging.info(f"사용자 상호작용 수 조회 시작 (user_id: {user_id})")
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM view_ftness_for_model WHERE user_index = %s", 
                (user_id,)
            )
            interaction_count = cursor.fetchone()[0]
            cursor.close()
            logging.info(f"사용자 상호작용 수: {interaction_count}")
            return interaction_count
        except Exception as e:
            logging.error(f"사용자 상호작용 수 조회 오류: {e}")
            return 0

    # 벡터 임베딩 생성
    async def create_embedding(self, text: str) -> np.ndarray:
        try:
            logging.info(f"임베딩 생성 시작: {text}")
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                text
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"임베딩 생성 오류: {e}")
            return np.zeros(Settings.VECTOR_DIMENSION, dtype=np.float32)

    # 대화 저장
    async def add_conversation(self, user_id: int, user_data: Dict, message: str) -> str:
        try:
            if not InputValidator.validate_message(message):
                raise ValueError("유효하지 않은 메시지입니다.")
            logging.info(f"대화 추가 시작 (user_id: {user_id})")

            rec_log = REC_LOG(
                user_index=user_id,
                messages=[{"prompt": user_data, "content": message}],
                reg_date=datetime.now(),
            )

            await self.save_conversation(rec_log)
            logging.info(f"대화 추가 완료 (user_id: {user_id})")
            return user_id
        except Exception as e:
            logging.error(f"대화 추가 오류: {e}")
    
    async def save_conversation(self, rec_log: REC_LOG):
        try:
            logging.info(f"대화 저장 시작 (user_index: {rec_log.user_index})")
            cursor = self.db_connection.cursor()
            cursor.execute(
                """INSERT INTO rec_log 
                (user_index, rec_log_context, reg_date) 
                VALUES (%s, %s, %s)""",
                (
                    rec_log.user_index, 
                    str(rec_log.messages), 
                    rec_log.reg_date
                )
            )
            self.db_connection.commit()
            cursor.close()
            logging.info(f"대화 저장 완료 (user_index: {rec_log.user_index})")
        except Exception as e:
            logging.error(f"대화 저장 오류: {e}")