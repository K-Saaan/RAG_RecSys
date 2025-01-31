import logging
import numpy as np
from typing import List, Dict, Tuple
import faiss
from langchain.embeddings import OpenAIEmbeddings
from core.memory_manager import ChatbotMemoryManager
from config.settings import Settings

# 로깅 설정
logging.basicConfig(
    filename="log_cf.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class CollaborativeFiltering:
    def __init__(self, settings, memory_manager):
        self.settings = settings
        self.memory_manager = memory_manager
        logging.info("CF Recommender 초기화 시작")

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Settings.OPENAI_API_KEY
        )
        
        self.user_vector = memory_manager.user_vector
        self.exercise_vector = memory_manager.exercise_vector
        self.interaction_vector = memory_manager.interaction_vector
        
        self.user_metadata = memory_manager.user_metadata
        self.exercise_metadata = memory_manager.exercise_metadata
        self.interaction_metadata = memory_manager.interaction_metadata

    
    async def recommend_for_user(self, user_id: int, similar_users: List[Dict], user_data: Dict, k: int = 5) -> List[str]:
        try:
            # 유사 사용자의 산호 운동 수집
            logging.info("유사 사용자 선호 운동 검색")
            similar_users_exercises = await self.get_similar_users_exercises(similar_users)
            logging.info("유사 사용자 선호 운동 검색 종료")
            
            # 추천 운동 계산
            logging.info("추천 운동 계산")
            recommended_exercises = await self.compute_collaborative_recommendations(
                user_id, 
                similar_users_exercises, 
                user_data, 
                k
            )
            logging.info("추천 운동 계산 종료")
            logging.info(f"recommended_exercises : {recommended_exercises}")
            return recommended_exercises
        
        except Exception as e:
            logging.error(f"협업 필터링 추천 오류: {e}")
            return []

    # 유사 사용자 수집    
    async def get_similar_users_exercises(self, similar_users: List[Dict]) -> Dict[str, List[str]]:
        similar_users_exercises = {}
        for user in similar_users:
            user_index = user.get('user_index')
            if user_index:
                logging.info(f"선호 데이터에서 사용자 조회 : {user_index}")
                exercises = await self.fetch_user_exercise_history(user_index)
                similar_users_exercises[user_index] = exercises
        
        return similar_users_exercises
    
    # 사용자의 운동기록 검색
    async def fetch_user_exercise_history(self, user_index: int) -> List[str]:
        user_interactions = [interaction for interaction in self.interaction_metadata if interaction['user_index']==user_index]
        logging.info(f"유사 사용자의 선호 데이터 : {user_interactions}")
        return user_interactions
    
    # 추천 운동 계산
    async def compute_collaborative_recommendations(
        self, 
        user_id: int, 
        similar_users_exercises: Dict[str, List[str]], 
        user_data: Dict, 
        k: int
    ) -> List[str]:
        
        # 유사 사용자의 선호도 확인
        logging.info("유사 사용자의 선호도 확인")
        logging.info(f"user_id : {user_id}")
        logging.info(f"user_data : {user_data}")
        logging.info(f"similar_users_exercises : {similar_users_exercises}")

        exercise_scores = {}
        for similar_user_id, exercises in similar_users_exercises.items():
            logging.info(f"similar_user_id: {similar_user_id}, exercises: {exercises}")
            for exercise in exercises:
                exercise_name = exercise.get('exercise_name')
                like_yn = exercise.get('like_yn', 0)
                # score = self.get_preference_score(similar_user_id, exercise_name)
                exercise_scores[exercise_name] = exercise_scores.get(exercise_name, 0) + like_yn
        
        logging.info("유사 사용자 선호도 계산 완료")

        # 현재 사용자의 선호도 추가
        current_user_exercises = [
            {
                "exercise_name": interaction["exercise_name"],
                "like_yn": interaction["like_yn"]
            }
            for interaction in self.interaction_metadata
            if interaction['user_index'] == user_id
        ]
        logging.info(f"현재 사용자 선호 운동 : {current_user_exercises}")

        for exercise in current_user_exercises:
            exercise_name = exercise.get('exercise_name')
            like_yn = exercise.get('like_yn', 0)
            
            # 현재 사용자의 선호도는 가중치를 더 높게 부여
            exercise_scores[exercise_name] = exercise_scores.get(exercise_name, 0) + like_yn * 2

        logging.info("현재 사용자 선호도 계산 완료")
        
        # 점수에 따라 정렬
        sorted_exercises = sorted(
            exercise_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        logging.info(f"정렬된 운동 점수: {sorted_exercises}")

        # 사용자 데이터 기반 운동 필터링
        # 중복 제거 및 상위 k개 선택
        unique_recommendations = []
        seen_exercises = set()

        for exercise, score in sorted_exercises:
            if exercise not in seen_exercises:
                unique_recommendations.append((exercise, score))
                seen_exercises.add(exercise)
            if len(unique_recommendations) >= k:
                break

        logging.info(f"중복 제거 후 상위 {k}개 추천 운동: {unique_recommendations}")

        # 사용자 데이터 기반 운동 필터링
        recommendations = self.filter_recommendations(
            unique_recommendations, 
            user_data, 
            k
        )
        logging.info(f"최종 추천 운동: {recommendations}")

        return [exercise for exercise, _ in recommendations]
    
    # 사용자의 운동 선호토 확인
    def get_preference_score(self, user_index: int, exercise: str) -> float:
        logging.info("사용자의 선호도 점수 확인")
        logging.info(f"user_index : {user_index}")
        logging.info(f"exercise : {exercise}")

        return self.interaction_metadata.get((user_index, exercise), 0)
    
    # 추천 운동을 사용자 데이터 기반으로 필터링
    def filter_recommendations(
        self, 
        sorted_exercises: List[Tuple[str, int]], 
        user_data: Dict, 
        k: int
    ) -> List[Tuple[str, int]]:
        
        filtered_recommendations = []
        
        for exercise, score in sorted_exercises:
            filtered_recommendations.append((exercise, score))
            if len(filtered_recommendations) == k:
                    break

        logging.info(f"필터링된 추천 목록 : {filtered_recommendations}")
        return filtered_recommendations