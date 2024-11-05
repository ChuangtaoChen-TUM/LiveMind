from live_mind.abc import BaseStreamModel
from collections.abc import Generator
from . import gradio, vllm_session

__all__ = ["gradio", "vllm_session"]

LLAMA_MODELS = ["llama-3-8b", "llama-3-70b"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
SUPPORTED_MODELS = LLAMA_MODELS + OPENAI_MODELS
def get_stream_model(name: str) -> BaseStreamModel:
    assert name in LLAMA_MODELS + OPENAI_MODELS, f"Model {name} is not supported."
    if name in LLAMA_MODELS:
        return _get_stream_vllm_model(name)
    elif name in OPENAI_MODELS:
        return _get_stream_openai_model(name)

def _get_stream_vllm_model(name: str) -> BaseStreamModel:
    return vllm_session.Session(name, temperature=0.0)

def _get_stream_openai_model(name: str) -> BaseStreamModel:
    import openai
    assert name in OPENAI_MODELS, f"Model {name} is not supported."
    client = openai.OpenAI()

    class Model(BaseStreamModel):
        def chat_complete(self, message: list[dict[str, str]]) -> str:
            response = client.chat.completions.create(
                model=name,
                messages=message,
                temperature=0,
            )
            response_text = response.choices[0].message.content
            if not response_text:
                response_text = ""
            return response_text
        
        def stream(self, message: list[dict[str, str]]) -> Generator[str, None, None]:
            print(message)
            stream = client.chat.completions.create(
                model=name,
                messages=message,
                temperature=0,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
    
    return Model()

