import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, List, Dict, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from config.settings import Settings
from utils.validators import InputValidator
from core.rag_recommender import RAGRecommender
from core.cf_recommender import CollaborativeFiltering

# 로깅 설정
logging.basicConfig(
    filename="log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class KoreanChatbot:
    def __init__(self):
        from core.memory_manager import ChatbotMemoryManager
        logging.info("챗봇 초기화 시작")
        self.memory_manager = ChatbotMemoryManager(Settings)
        self.rag_recommender = RAGRecommender(Settings, self.memory_manager)
        self.cf_recommender = CollaborativeFiltering(Settings, self.memory_manager)
        logging.info("챗봇 초기화 완료")
        
        self.llm = ChatOpenAI(
            model_name=Settings.CHAT_MODEL,
            temperature=Settings.TEMPERATURE,
            max_tokens=Settings.MAX_TOKENS,
            openai_api_key=Settings.OPENAI_API_KEY
        )
        
        # Interaction threshold 설정
        self.interaction_threshold = Settings.DEFAULT_INTERACTION_THRESHOLD
        
        # 프롬프트 템플릿 설정
        self.prompt_template = recommendation_prompt_template = PromptTemplate(
            input_variables=["user_profile", "recommended_exercises"],
            template="""
            You are a professional and empathetic exercise recommendation assistant. Your goal is to help users improve their health and well-being by providing clear, friendly, and personalized exercise recommendations.

            Below are the user’s profile information and a list of recommended exercises. Using this information:
            1. Explain why each recommended exercise is suitable for the user, considering their age, gender, disability type, disability grade, and fitness goals.
            2. Provide detailed, step-by-step instructions on how to perform each exercise safely and effectively.
            3. Include any necessary precautions or adaptations to ensure the exercises are accessible and beneficial for the user.

            Use encouraging and supportive language to motivate the user. Write your response in Korean, using the structure below:

            [USER PROFILE]
            {user_profile}

            [RECOMMENDED EXERCISES]
            {recommended_exercises}

            [ANSWER FORM]
            운동명: [Exercise Name]
            - 추천 이유: Explain why this exercise is ideal for the user. Include specific details like how it aligns with their fitness goals, supports their disability needs, or benefits their age and gender.
            - 수행 방법: Provide step-by-step instructions for performing the exercise. Mention proper posture, technique, repetitions, and breathing tips.
            - 주의 사항: Suggest any precautions or modifications the user should consider based on their disability type or grade.

            Write an answer for each recommended exercise listed in {recommended_exercises}, following the format and instructions above.

            [EXAMPLE RESPONSE]
            운동명: 스쿼트
            - 추천 이유: 스쿼트는 하체 근력을 강화하고 안정성을 높이는 데 효과적입니다. 사용자의 현재 체력 수준과 장애 유형을 고려할 때, 무릎과 허리 부담을 줄이면서 수행할 수 있는 적절한 운동입니다.
            - 수행 방법: 어깨 너비로 다리를 벌리고 서서 허리를 곧게 세우세요. 천천히 무릎을 굽히며 엉덩이를 뒤로 빼고, 허벅지가 바닥과 평행할 때까지 내립니다. 그 후, 천천히 원래 자세로 돌아옵니다. 12~15회 반복하세요.
            - 주의 사항: 허리를 과도하게 구부리지 않도록 주의하세요. 무릎에 통증이 있는 경우 얕은 자세로 진행하거나 전문가의 조언을 받으세요.

            Follow this format for each recommended exercise.
            """,
        )

    async def periodic_data_refresh(self, interval: int):
        """주기적으로 데이터를 갱신"""
        while True:
            logging.info("주기적 데이터 갱신 시작")
            try:
                await self.memory_manager.preload_data()
                logging.info("주기적 데이터 갱신 완료")
            except Exception as e:
                logging.error(f"데이터 갱신 중 오류 발생: {e}")
            await asyncio.sleep(interval)  # interval 초마다 반복

    async def async_initialize(self):
        """
        비동기 초기화를 수행하는 메서드.
        """
        try:
            logging.info("챗봇 비동기 초기화 시작")
            await self.memory_manager.create_connection_pool()
            await self.memory_manager.preload_data()
            logging.info("챗봇 비동기 초기화 완료")
        except Exception as e:
            logging.error(f"챗봇 비동기 초기화 중 오류 발생: {e}")

    # 사용자 입력 기반 추천 메시지 생성
    async def process_message(self, user_data: Dict) -> str:
        user_id = int(user_data['user_index'])
        logging.debug(f"process_message 호출 - user_id: {user_id}, user_data: {user_data}")
        
        if not InputValidator.validate_user_id(user_id):
            logging.error("유효하지 않은 사용자 ID")
            raise ValueError("유효하지 않은 사용자 ID입니다.")

        try:
            logging.info("Process 1")
            logging.info(f"사용자 ID {user_id}에 대한 상호작용 횟수 조회 시작")
            interaction_count = await self.memory_manager.get_user_interaction_count(user_id)
            logging.info(f"사용자 ID {user_id}의 상호작용 횟수: {interaction_count}")
            
            logging.info("Process 2")
            if interaction_count < Settings.DEFAULT_INTERACTION_THRESHOLD:
                logging.info("RAG 기반 추천 생성 시작")
                recommendations = await self.get_rag_recommendations(user_data)
                logging.info(f"RAG 기반 추천 결과: {recommendations}")
            else:
                logging.info("CF 기반 추천 생성 시작")
                recommendations = await self.get_collaborative_recommendations(user_id, user_data)
                logging.info(f"CF 기반 추천 결과: {recommendations}")
            
            logging.info("Process 3")
            logging.info("LLM을 사용하여 개인화된 응답 생성 시작")
            response = await self.generate_personalized_response(user_data, recommendations)
            logging.info(f"생성된 응답: {response}")
            
            logging.info("Process 4")
            logging.info(f"사용자 ID {user_id}의 추천 내용을 메모리에 저장")
            await self.memory_manager.add_conversation(user_id, user_data, response)
            
            return response

        except Exception as e:
            logging.error(f"메시지 처리 오류: {e}")
            return "죄송합니다. 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    async def generate_personalized_response(self, user_data: Dict, recommendations: List[str]) -> str:
        
        logging.info(f"응답 생성 >>>>>>>>>>>. ")
        logging.info(f"사용자 프로필 : {user_data}")
        logging.info(f"사용자 이름 : {user_data['name']}")
        logging.info(f"추천 운동 리스트 : {recommendations}, {type(recommendations)}")
        
        try:
            # 사용자 프로필 생성
            user_profile_text = (
                f"이름: {user_data['name']}\n"
                f"나이: {user_data['age']}세\n"
                f"성별: {user_data['gender']}\n"
                f"장애 유형: {user_data['disability_type']}\n"
                f"장애 등급: {user_data['disability_grade']}\n"
                f"운동 목표: {user_data['exercise_goal']}\n"
                f"운동 경험: {user_data['exercise_experience']}"
            )
            logging.info(f"user_profile_text : {user_profile_text}")
            # 추천 운동 텍스트 포맷
            if not recommendations or not isinstance(recommendations, list):
                logging.error("추천 운동이 None이거나 리스트가 아닙니다.")
                recommendations = ["AI 추천 운동이 없습니다. 다시 시도해주세요."]
            
            logging.info(f"프롬프트 생성 >>>>>>>>>>>. ")

            # 프롬프트 생성
            prompt = self.prompt_template.format(
                user_profile=user_profile_text,
                recommended_exercises=recommendations
            )
            logging.info(f"생성된 프롬프트: {prompt}")
            
            # LLM을 사용하여 답변 생성
            response = await self.llm.apredict(prompt)
            return response
                
        except Exception as e:
            logging.error(f"응답 생성 오류: {e}")
            return "죄송합니다. 개인화된 운동 추천을 생성하는 중에 오류가 발생했습니다."
    
    # Cold-Scenario
    # RAG 기반 추천 및 답변 생성
    async def get_rag_recommendations(self, user_data: Dict) -> Tuple[List[str], List[str]]:
        # RAG 운동 추천
        logging.info("RAG 운동 추천 시작")
        recommendations = await self.rag_recommender.retrieve_recommendations(
            user_data, k=3, use_mmr=True
        )
        logging.info(f"추천 운동 : {recommendations}")
        
        return recommendations
    
    # Warm-Scenario
    # 협업 필터링을 사용한 운동 추천 및 답변 생성
    async def get_collaborative_recommendations(self, user_id: int, user_data: Dict) -> Tuple[List[str], List[str]]:
        logging.info("CF 운동 추천 시작 >>>>>>>>>>>>>>> ")
        # 유사 사용자 검색
        logging.info("유사 사용자 검색 시작")
        similar_users = await self.rag_recommender.retrieve_similar_users(user_id, user_data)
        logging.info(f"유사 사용자 수 : {len(similar_users)}")

        # 협업 필터링 수행
        logging.info("CF 추천 시작")
        recommendations = await self.cf_recommender.recommend_for_user(
            user_id, similar_users, user_data, k=3
        )
     
        return recommendations
    
    