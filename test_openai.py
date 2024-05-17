"""
Testing deploying a local OpenAI API with the litellm package.
"""
import time
from fastapi import FastAPI
from typing import List, Optional, Union
from pydantic import BaseModel
from litellm import completion

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

app = FastAPI(title="OpenAI-compatible API")

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    tool_calls = ChatCompletionMessageToolCall(id='1',
        function=Function(name="Response", arguments='{"name": "test", "description": "test"}'), 
                                               args=[])
    message = ChatMessage(role="user", content="Say this is a test", tool_calls=[tool_calls])
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": message,
        }],
    }

if __name__ == '__main__':
    import instructor 
    from litellm import completion

    # init client and connect to localhost server
    client = instructor.from_litellm(completion)

    # call API
    response = client.chat.completions.create(messages=[
            {
                "role": "user",
                "content": "Say this is a test",
                "tool_calls": [] # this is weirdly required or it fails. Why? 
            }
        ],
        response_model=Response,
        model="gpt-4-turbo",
        base_url="http://localhost:8000",
    )
    print(response)
