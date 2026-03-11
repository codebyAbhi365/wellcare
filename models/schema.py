# models/schemas.py
from pydantic import BaseModel
from typing import Optional

class AlertRequest(BaseModel):
    user_id: str
    meal_logged: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str