from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

@dataclass
class REC_LOG:
    user_index: int
    messages: List[Dict]
    reg_date: datetime
    
    def to_dict(self) -> Dict:
        return {
            'user_index': self.user_index,
            'messages': self.messages,
            'reg_date': self.reg_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'REC_LOG':
        data['reg_date'] = datetime.fromisoformat(data['reg_date'])
        return cls(**data)