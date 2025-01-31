import streamlit as st
from core.chatbot import KoreanChatbot
import asyncio

# page 설정
def set_page_config():
    st.set_page_config(
        page_title="RAG RecSys",
        page_icon="🧑‍🦽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# header 설정
def set_page_header():
    st.header("맞춤형 운동 추천 서비스", anchor=False)

set_page_config()
set_page_header()

# 캐시로 챗봇 호출
@st.cache_resource
def get_chatbot() -> KoreanChatbot:
    return KoreanChatbot()

# 사용자 정보 입력 함수
def get_user_input():
    st.subheader("회원 정보를 입력해주세요")
    name = st.text_input("이름")
    user_index = st.text_input("사용자 번호")
    age = st.number_input("나이", min_value=1, max_value=120, step=1)
    gender = st.selectbox("성별", ["남성", "여성"])
    disability_type = st.selectbox(
        "장애 유형", ["시각장애", "청각장애", "지적장애", "척수장애"]
    )
    disability_grade = st.selectbox("장애 등급", ["1등급", "2등급", "3등급"])
    exercise_goal = st.text_input("운동 목표 (예: 체력 향상, 유연성 강화 등)")
    exercise_experience = st.text_input("운동 경험 (예: 완전 없음, 거의 없음, 보통, 많음, 매우 많음)")
    
    return {
        "name": name,
        "user_index":user_index,
        "age": age,
        "gender": gender,
        "disability_type": disability_type,
        "disability_grade": disability_grade,
        "exercise_goal": exercise_goal,
        "exercise_experience": exercise_experience,
    }

# 서비스 시작 시 한 번만 호출되는 글로벌 초기화 함수 추가
async def initialize_chatbot():
    chatbot = get_chatbot()
    await chatbot.async_initialize()
    return chatbot

# Streamlit 앱 초기화 시 호출
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = asyncio.run(initialize_chatbot())

async def main():
    chatbot = st.session_state.chatbot
    # 사용자 정보 입력받기
    user_info = get_user_input()
    
    # 모든 필드가 입력되었는지 확인
    if st.button("추천받기"):
        if all(user_info.values()):  # 모든 값이 입력되었을 때만 처리
            with st.spinner("추천을 생성 중입니다..."):
                # 사용자 정보를 챗봇 프로세스 함수에 전달
                response = await chatbot.process_message(user_info)
                
                # 응답을 화면 하단에 출력
                st.subheader("추천 결과")
                st.write(f"챗봇 추천 운동: {response}")
        else:
            st.warning("모든 정보를 입력해주세요.")
        

if __name__ == "__main__":
    asyncio.run(main())