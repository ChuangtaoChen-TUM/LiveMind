""" Configuration file """
from abc import ABC, abstractmethod
# path to the MMLU-PRO dataset
# /data2/NNdata/dataset/mmlu-pro/data/data/
MMLU_PRO_PATH = "/data2/NNdata/dataset/mmlu-pro/data/data/"
# path to the GSM-8k dataset
GSM8K_PATH = "/data2/NNdata/dataset/gsm8k/data/main"
# config the model path if you use 'get_model_vllm_example'
LLAMA_3_8B_PATH = "/data2/NNdata/model_file/llama3/llama3_8b_instruct_awq/model/" # replace this with your own path
LLAMA_3_70B_PATH = "/data2/NNdata/model_file/llama3/llama3_70b_instruct_awq/model/" # replace this with your own path
# /data2/NNdata/model_file/llama3/llama3_70b_instruct_awq/model/
LLAMA_MODELS = ["llama-3-8b", "llama-3-70b"]
OPENAI_MODELS = ["got-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet"]
"""
The get model function should return a model with the following methods:
- chat_complete: takes a message and returns a response
the response format is similar to OpenAI's API.

for example:
model = get_model("llama-3-8b")
message = [
    {
        "role": "system",
        "content": "This is a system message."
    },
    {
        "role": "user",
        "content": "This is message 2."
    }
]
response: dict = model.chat_complete(message)

the response is a dict and should have the following format:
{
    "choices": [
        {
            "message": {
                "content": "This is a response message."
            }
        }
    ],
    "usage": {
        "completion_tokens": 20,
        "prompt_tokens": 30,
        "total_tokens": 50
    }
}
"""

def get_model(name: str):
    assert name in LLAMA_MODELS + OPENAI_MODELS, f"Model {name} is not supported."
    if name in LLAMA_MODELS:
        print("using get_model_vllm_example as the model function, you can replace this with your own implementation.")
        return get_model_vllm_example(name) # You can also use this example implementation
    # elif name in OPENAI_MODELS:
    #     print("using get_model_openai_example as the model function, you can replace this with your own implementation.")
    #     return get_model_openai_example(name)

def get_model_vllm_example(name: str) -> 'BaseModel':
    """ This is an example implementation of the model function using VLLM. You can replace this with your own implementation.
    To use this function, you need to install the following packages:
    - vllm
    - transformers
    Besides, you need to download the model files and set the path in the config.py file.
    """
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer
    model_path_dict = {
        "llama-3-8b": LLAMA_3_8B_PATH,
        "llama-3-70b": LLAMA_3_70B_PATH
    }
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.9
    model_path = model_path_dict[name]
    if model_path == "":
        raise ValueError(f"Please set the model path for {name} in the config.py file")
    model = LLM(
        model_path, model_path, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    param = SamplingParams(
        temperature=0,
        max_tokens=8192,
        top_p=0.9,
        stop_token_ids=stop_token_ids,
    )
    class Model(BaseModel):
        def chat_complete(self, message: list[dict]) -> str:
            prompts = tokenizer.apply_chat_template(
                [message,],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer.batch_encode_plus(
                prompts
            )["input_ids"]
            input = inputs[0]
            vllm_results = model.generate(
                prompt_token_ids=inputs,
                sampling_params=param,
                use_tqdm=False
            )
            vllm_result = vllm_results[0]
            result = vllm_result.prompt_token_ids+vllm_result.outputs[0].token_ids
            # remove the input tokens from the output
            assert len(result) >= len(input)
            assert result[:len(input)] == input
            result = result[len(input):]
            if len(result) > 0 and result[-1] in stop_token_ids:
                result = result[:-1]

            generated_texts = tokenizer.batch_decode(
                [result,],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False
            )
            return generated_texts[0]

    model_with_chat_complete = Model()
    return model_with_chat_complete


# def get_model_openai_example(name: str) -> 'BaseModel':
#     import openai
#     assert name in OPENAI_MODELS, f"Model {name} is not supported."
#     client = openai.OpenAI()

#     class Model(BaseModel):
#         def chat_complete(self, message):
#             response = client.chat.completions.create(
#                 model=name,
#                 messages=message,
#                 temperature=0,
#             )
#             return response
        
#     model_with_chat_complete = Model()
#     return model_with_chat_complete

# def get_model_anthropic_example(name: str) -> 'BaseModel':
#     import anthropic
#     assert name in ANTHROPIC_MODELS, f"Model {name} is not supported."
#     model_name_dict = {
#         "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
#         "claude-3-opus": "claude-3-opus-20240229",
#         "claude-3-sonnet": "claude-3-sonnet-20240229"
#     }
#     client = anthropic.Anthropic()
    
#     class Model(BaseModel):
#         def chat_complete(self, message):
#             system_message, message = self._convert_to_anthropic_format(message)
#             if system_message == "":
#                 system_message = anthropic.NOT_GIVEN
#             response = client.messages.create(
#                 model=model_name_dict[name],
#                 system=system_message,
#                 messages=message,
#                 temperature=0,
#                 max_tokens=4096
#             )
#             content = response.content[0].text
#             usage = response.usage
#             output = {
#                 "choices": [{"message": {"content": content, "role": "assistant"}}],
#                 "usage": {
#                     "completion_tokens": usage.output_tokens,
#                     "prompt_tokens": usage.input_tokens,
#                     "total_tokens": usage.input_tokens + usage.output_tokens
#                 }
#             }
#             return output
        
#         @staticmethod
#         def _convert_to_anthropic_format(message: list[dict[str, str]]):
#             system_message = ""
#             formatted_message = []
#             for m in message:
#                 if m["role"] == "system":
#                     system_message += m["content"]
#                 else:
#                     formatted_message.append({
#                         "role": m["role"],
#                         "content": [{"type": "text", "text": m["content"]}]
#                     })
#             return system_message, formatted_message
    
#     model_with_chat_complete = Model()
#     return model_with_chat_complete


class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
        pass
