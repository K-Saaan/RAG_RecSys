import pytest
import asyncio
from datetime import datetime, timedelta
from core.chatbot import KoreanChatbot

@pytest.mark.asyncio
async def test_chatbot_response():
    chatbot = KoreanChatbot()
    test_user_id = "test_user_123"
    test_message = "안녕하세요!"
    
    response = await chatbot.process_message(test_user_id, test_message)
    assert response and isinstance(response, str)

@pytest.mark.asyncio
async def test_auto_conversation_initiation():
    chatbot = KoreanChatbot()
    test_user_id = "test_user_123"
    
    # 마지막 상호작용 시간을 임계값보다 이전으로 설정
    chatbot.memory_manager.last_interaction[test_user_id] = (
        datetime.now() - timedelta(minutes=31)
    )
    
    auto_message = await chatbot.check_and_initiate_conversation(test_user_id)
    assert auto_message and isinstance(auto_message, str)