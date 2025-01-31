# RAG_RecSys


## 프로젝트 소개

최근 LLM을 활용한 추천 시스템 연구가 활발해지고 있다.[Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System 등] LLM을 Fine-tuning하거나 상품 정보를 LLM 토큰과 align하는 것은 많은 자원을 필요로 한다. <br>
추천 시스템을 설계할 때 사용하는 데이터가 뉴스와 같이 텍스트 데이터가 주인 경우 RAG를 적용한 추천 시스템이 더 효율적일 수 있다. <br>
추가적으로 MMR을 적용해 추천의 다양성을 확보한다면 사용자의 serendipity도 기대할 수 있을 것이다. <br>
참고 : [RAGSys: Item-Cold-Start Recommender as RAG System](https://arxiv.org/pdf/2405.17587) 

---

## 사용 데이터
- 국민체육진흥공단
> 장애인 체력측정별 추천 운동 데이터
> 장애인 스포츠 강좌 이용권 시설 정보
> 장애인 스포츠 강좌 이용권 강좌 정보 
> 장애인 체력 측정별 운동처방 데이터

---

## 서비스 흐름도

<img width="835" alt="Image" src="https://github.com/user-attachments/assets/f7c797b4-771c-44e4-ae39-cbaa62ceb5f3" />

1. User Profile 정보 입력
2. 해당 User의 Interaction 횟수 확인
3. 5회 이하인 경우 Contents 기반 추천 (RAG만 사용)
4. 5회 초과인 경우 CF(RAG기반 CF) 기반 추천
5. 추천 결과에 따라 LLM으로 출력값 생성

### Cold-Scenario

<img width="779" alt="Image" src="https://github.com/user-attachments/assets/13e1e458-92f8-490c-87d2-b6d7738693a9" />

1. User Profile embedding
2. User vector DB에서 유사한 user 추출
3. 유사 user들의 운동 데이터 조회 (사용 데이터에 존재하는 데이터 활용)
4. MMR 추천 수행
5. 최종 운동 반환


### Warm-Scenario

<img width="877" alt="Image" src="https://github.com/user-attachments/assets/d36e9b01-49b3-4009-9b4d-c0693ce80f64" />

1. User Profile embedding
2. User vector DB에서 유사한 user 추출
3. 유사 user들의 운동 데이터 조회 (사용 데이터에 존재하는 데이터 활용)
4. 사용자 선호 운동 조회
5. 유사 사용자 운동과 사용자 운동 각각 점수 계산(사용자 선호 운동에 더 높은 점수 부여)
6. 점수에 따라 정렬
7. 최종 운동 반환


### LLM
User Profile 정보와 반환된 운동을 받아 사용자에 따라 어떻게 운동을 수행하면 좋을지 가이드 생성


## 결론
딥러닝 기반의 기존 추천 시스템과 LLM 기반 추천 시스템과 다른 RAG만을 사용해 상황에 따른 하이드리드 추천 서비스 개발. <br>
추후 벤치마크 실험을 진행하여 실제 성능과 효율성 확인 진행 예정