from datetime import datetime
from typing import Any, Dict

# 유효성 검사
class InputValidator:
    @staticmethod
    def validate_message(message: str) -> bool:
        return bool(message and message.strip())
    
    @staticmethod
    def validate_user_id(user_id: int) -> bool:
        return bool(user_id and user_id)
    
    @staticmethod
    def validate_conversation(conversation: Dict[str, Any]) -> bool:
        required_fields = ['id', 'user_id', 'messages', 'timestamp']
        return all(field in conversation for field in required_fields)