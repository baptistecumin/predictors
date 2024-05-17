"""
Mocking an OpenAI API with function calling.
"""
from typing import List, Optional, Union
from pydantic import BaseModel

class Function(BaseModel):
    name: str
    arguments: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function
    args: List

class ChatMessage(BaseModel):
    role: str
    content: str
    tool_calls: Union[None, List[ChatCompletionMessageToolCall]]

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    tool_calls: List[ChatCompletionMessageToolCall] = []
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class Response(BaseModel):
    name: str
    description: str
