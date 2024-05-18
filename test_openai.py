"""
Testing deploying a local OpenAI API with the litellm package.
"""
import time
import json
from fastapi import FastAPI
from litellm import completion
from openai_models import ChatCompletionRequest, ChatCompletionMessageToolCall, ChatMessage, Function, Response

app = FastAPI(title="OpenAI-compatible API")

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # do inference 
    results_dict = {"name": "test", "description": "test"}
    json_results_dict = json.dumps(results_dict)
    tool_calls = ChatCompletionMessageToolCall(id='1',
        function=Function(name="Response", arguments=json_results_dict), 
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
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1

    client = instructor.from_litellm(completion)
    response = client.chat.completions.create(messages=[
            {
                "role": "user",
                "content": "Say this is a test",
                "tool_calls": [] # this is weirdly required or it fails. Why? 
            }
        ],
        response_model=Response,
        model="gpt-4-turbo",
        #base_url="http://localhost:8000",
        base_url='https://baptistecumin--train-peft-unslothfinetunedclassifier-inf-947252.modal.run'
    )
    print(response)
