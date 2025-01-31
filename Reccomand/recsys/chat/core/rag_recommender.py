import logging
import numpy as np
from typing import List, Dict, Optional
from langchain.embeddings import OpenAIEmbeddings
from core.memory_manager import ChatbotMemoryManager
import faiss

from config.settings import Settings

# 로깅 설정
logging.basicConfig(
    filename="log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class RAGRecommender:
    def __init__(self, settings, memory_manager):
        self.settings = settings
        self.memory_manager = memory_manager
        logging.info("RAG Recommender 초기화 시작")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Settings.OPENAI_API_KEY
        )
        
        self.user_vector = memory_manager.user_vector
        self.exercise_vector = memory_manager.exercise_vector
        
        self.user_metadata = memory_manager.user_metadata
        self.exercise_metadata = memory_manager.exercise_metadata
        logging.info(f"초기 사용자 메타 데이터 :{self.user_metadata}\n")
        logging.info(f"초기 운동 메타 데이터 : {self.exercise_metadata}\n")
        logging.info("FAISS 인덱스 초기화 완료")
        
   
    # 추천 운동 검색
    async def retrieve_recommendations(self, user_data: Dict, k: int, use_mmr: bool = True) -> List[str]:
        try:
            logging.info("추천 운동 검색 시작")
            user_vector = await self.create_user_embedding(user_data)
            logging.info(f"사용자 임베딩 생성 완료: {user_vector.shape}")
            
            if use_mmr:
                logging.info("MMR 기반 추천 수행")
                recommendations = await self.mmr_recommendation(user_vector, k)
            else:
                logging.info("단순 유사도 기반 추천 수행")
                recommendations = await self.simple_recommendation(user_vector, k)
            
            logging.info(f"추천 결과: {recommendations}")
            return recommendations
        except Exception as e:
            logging.error(f"추천 생성 중 오류 발생: {e}")
            return []
    
    # 유사 사용자 검색
    async def retrieve_similar_users(self, user_id, user_data: Dict, k: int = 5) -> List[Dict]:
        try:
            logging.info("유사 사용자 검색 시작")
            user_vector = await self.create_user_embedding(user_data)
            logging.info(f"사용자 임베딩 생성 완료: {user_vector.shape}")
             
             # 사용자 인덱스에서 유사 사용자 검색
            logging.info("사용자 인덱스에서 유사 사용자 검색")
            distances, indices = self.user_vector.search(
                user_vector.reshape(1, -1), 
                k * 5
            )
            
            # 검색된 유사 사용자 프로필 반환
            logging.info("검색된 유사 사용자 정보 조회")
            similar_users = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.user_metadata):
                    similar_users.append(self.user_metadata[idx])
            
            logging.info(f"조회된 유사 사용자 : {similar_users}")
            return similar_users
        except Exception as e:
            logging.error(f"추천 생성 중 오류 발생: {e}")
            return {}
    
    # 사용자 프로필 데이터 벡터 임베딩
    async def create_user_embedding(self, user_data: Dict) -> np.ndarray:
        try:
            user_text = self.convert_user_data_to_text(user_data)
            logging.info(f"사용자 데이터를 텍스트로 변환: {user_text}")
            
            embedding = await self.embeddings.aembed_query(user_text)
            logging.info(f"사용자 임베딩 생성 완료")
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"사용자 임베딩 생성 중 오류 발생: {e}")
            return np.array([])
    
    # 사용자 프로필 데이터 -> 텍스트
    def convert_user_data_to_text(self, user_data: Dict) -> str:
        text = (
            f"장애 유형: {user_data.get('disability_type', 'Unknown')} "
            f"장애 등급: {user_data.get('disability_grade', 'Unknown')} "
            f"운동 경험: {user_data.get('exercise_experience', 'Unknown')} "
            f"운동 목표: {user_data.get('exercise_goal', 'Unknown')} "
            f"나이: {user_data.get('age', 'Unknown')} "
        )
        logging.info(f"사용자 프로필을 텍스트로 변환 완료: {text}")
        return text
    
    # MMR 추천
    async def mmr_recommendation(self, user_vector: np.ndarray, k: int, diversity_factor: float = 0.5) -> List[Dict[str, str]]:
        try:
            logging.info(f"MMR 추천을 시작")
            logging.info(f"운동 메타 데이터: {self.exercise_vector}")
            distances, initial_indices = self.exercise_vector.search(
                user_vector.reshape(1, -1), 
                k * 3
            )
            logging.info(f"초기 후보군 검색 완료. 후보 개수: {len(initial_indices[0])}")
            logging.info(initial_indices[0])

            initial_candidates = [
                (idx, distances[0][i])
                for i, idx in enumerate(initial_indices[0])
                if idx != -1
            ]
            logging.info(f'initial_candidates : {initial_candidates}')
            selected = []
            candidate_pool = initial_candidates.copy()
            while len(selected) < k and candidate_pool:
                best_candidate = None
                max_score = -np.inf
                
                logging.info('코사인 유사도 검색 시작')
                logging.info(f"FAISS 인덱스 내 데이터 개수: {self.exercise_vector.ntotal}")

                for candidate_idx, relevance in candidate_pool:
                    logging.info(f'candidate_idx : {candidate_idx}')
                    logging.info(f'relevance : {relevance}')
                    logging.info(f'selected : {selected}')
                    
                    for selected_idx, _ in selected:
                        logging.info(f'selected_idx : {selected_idx}')
                    
                    diversity_score = sum(
                        self.cosine_similarity(
                            self.exercise_vector.reconstruct(int(candidate_idx)),
                            self.exercise_vector.reconstruct(int(selected_idx))
                        )
                        for selected_idx, _ in selected
                    ) if selected else 0

                    mmr_score = diversity_factor * relevance - (1 - diversity_factor) * diversity_score
                    if mmr_score > max_score:
                        max_score = mmr_score
                        best_candidate = (candidate_idx, relevance)
                logging.info(f'best_candidate : {best_candidate}')
                if best_candidate:
                    selected.append(best_candidate)
                    candidate_pool.remove(best_candidate)
            logging.info(f'best_candidate 종료')

            logging.info(f'selected : {selected}')
            logging.info(f'exercise_metadata len: {len(self.exercise_metadata)}')
            
            for idx, score in selected:
                logging.info(f'selected idx : {idx}')
                logging.info(f'selected score : {score}')
            recommendations = [
                {
                    self.exercise_metadata[idx]['exercise_name']
                }
                for idx, _ in selected
            ]

            # 중복 제거 (인덱스를 기준으로)
            seen = set()
            unique_candidates = []
            for idx, relevance in initial_candidates:
                if idx not in seen:
                    unique_candidates.append((idx, relevance))
                    seen.add(idx)
            
            logging.info(f"Unique candidates: {unique_candidates}")
            
            # 점수 기준으로 정렬 후 상위 k개 선택
            top_k_candidates = sorted(unique_candidates, key=lambda x: x[1], reverse=True)[:k]
            logging.info(f"Top {k} candidates: {top_k_candidates}")
            
            # 추천 결과 생성
            recommendations = [
                {
                    "exercise_name": self.exercise_metadata[idx]['exercise_name']
                }
                for idx, _ in top_k_candidates
            ]
            logging.info(f"추천 완료. 추천 개수: {len(recommendations)}")
            return recommendations
            # logging.info(f"MMR 추천 완료. 추천 개수: {len(recommendations)}")
            # return recommendations
        except Exception as e:
            logging.error(f"MMR 추천 중 오류 발생: {e}")
            return []
    
    # 코사인 유사도 계산
    def cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        logging.info(f"코사인 유사도 계산 수행 >>>>>>>>>>> ")
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        similarity = dot_product / (norm_a * norm_b)
        logging.info(f"코사인 유사도 계산: {similarity}")
        return similarity