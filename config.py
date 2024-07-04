""" Configuration file """
from abc import ABC, abstractmethod
from typing import TypedDict
# path to the MMLU-PRO dataset
# /data2/NNdata/dataset/mmlu-pro/data/data/
MMLU_PRO_PATH = ""
# config the model path if you use 'get_model_vllm_example'
LLAMA_3_8B_PATH = "" # replace this with your own path
LLAMA_3_70B_PATH = "" # replace this with your own path
# /data2/NNdata/model_file/llama3/llama3_70b_instruct_awq/model/
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
    assert name in ["llama-3-8b", "llama-3-70b"]
    from chat_api import Session
    name_dict = {
        "llama-3-8b": "llama-3-8b-instruct-awq",
        "llama-3-70b": "llama-3-70b-instruct-awq"
    }
    session = Session(name_dict[name], backend="autoawq", output_format="streaming")
    class Model:
        def chat_complete(self, message):
            return session.chat_complete(message, result_format="openai")
    return Model()
    print("using get_model_vllm_example as the model function, you can replace this with your own implementation.")
    return get_model_vllm_example(name) # You can also use this example implementation

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
    class Model:
        def chat_complete(self, message):
            prompts = tokenizer.apply_chat_template(
                [message,],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer.batch_encode_plus(
                prompts
            )["input_ids"]
            input = inputs[0]
            prompt_len = len(inputs[0])
            vllm_results = model.generate(prompt_token_ids=inputs, sampling_params=param, use_tqdm=False)
            vllm_result = vllm_results[0]
            result = vllm_result.prompt_token_ids+vllm_result.outputs[0].token_ids
            total_len = len(result)
            gen_len = total_len - prompt_len
            # remove the input tokens from the output
            assert len(result) >= len(input)
            assert result[:len(input)] == input
            result = result[len(input):]
            if result[-1] in stop_token_ids:
                result = result[:-1]

            generated_texts = tokenizer.batch_decode(
                [result,],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False
            )
            output = {
                "choices": [{"message": {"content": generated_texts[0], "role": "assistant"}}],
                "usage": {
                    "completion_tokens": gen_len,
                    "prompt_tokens": prompt_len,
                    "total_tokens": total_len
                }
            } 
            return output

    model_with_chat_complete = Model()
    return model_with_chat_complete


class Response(TypedDict):
    """ Response from the model """
    choices: list[dict[str, dict[str, str]]]
    usage: dict[str, int]


class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> Response:
        pass
