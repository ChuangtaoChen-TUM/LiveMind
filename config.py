""" Configuration file """
from abc import ABC, abstractmethod

# the dataset path should contain the `.parquet` files
# path to the MMLU-PRO dataset
MMLU_PRO_PATH = ""
# path to the MMLU dataset
MMLU_PATH = ""
# config the model path if you use 'get_model_vllm_example'
# the model path should contain a `config.json` file
LLAMA_3_8B_PATH = "" # replace this with your own path
LLAMA_3_70B_PATH = "" # replace this with your own path

LLAMA_MODELS = ["llama-3-8b", "llama-3-70b"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet"]

def get_model(name: str):
    assert name in LLAMA_MODELS + OPENAI_MODELS + ANTHROPIC_MODELS, f"Model {name} is not supported."
    if name in LLAMA_MODELS:
        print("using get_model_vllm_example as the model function, you can replace this with your own implementation.")
        return get_model_vllm_example(name) # You can also use this example implementation
    elif name in OPENAI_MODELS:
        print("using get_model_openai_example as the model function, you can replace this with your own implementation.")
        return get_model_openai_example(name)
    elif name in ANTHROPIC_MODELS:
        print("using get_model_anthropic_example as the model function, you can replace this with your own implementation.")
        return get_model_anthropic_example(name)


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
    gpu_memory_utilization = 0.6 if name == "llama-3-70b" else 0.3
    model_path = model_path_dict[name]
    if model_path == "":
        raise ValueError(f"Please set the model path for {name} in the config.py file")
    model = LLM(
        model_path, model_path, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    max_tokens = 8192
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    param = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    class Model(BaseModel):
        def chat_complete(self, message: list[dict]) -> str:
            assert len(message) == 0 or message[-1]["role"] == "user"
            prompts = tokenizer.apply_chat_template(
                [message,],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts[0] = prompts[0]
            inputs = tokenizer.batch_encode_plus(
                prompts
            )["input_ids"]
            input = inputs[0]

            if len(input) > max_tokens:
                raise ValueError

            vllm_results = model.generate(
                prompt_token_ids=inputs,
                sampling_params=param,
                use_tqdm=False
            )
            vllm_result = vllm_results[0]
            result = vllm_result.prompt_token_ids+list(vllm_result.outputs[0].token_ids)
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


def get_model_openai_example(name: str) -> 'BaseModel':
    import openai
    assert name in OPENAI_MODELS, f"Model {name} is not supported."
    client = openai.OpenAI()

    if name == "gpt-4o":
        name = "gpt-4o-2024-05-13"
        print("setting the model to gpt-4o-2024-05-13")
    class Model(BaseModel):

        def chat_complete(self, message):
            response = client.chat.completions.create(
                model=name,
                messages=message,
                temperature=0,
            )
            response_text = response.choices[0].message.content
            if not response_text:
                response_text = ""
            return response_text

    model_with_chat_complete = Model()
    return model_with_chat_complete

def get_model_anthropic_example(name: str) -> 'BaseModel':
    # this has not been tested
    import anthropic
    assert name in ANTHROPIC_MODELS, f"Model {name} is not supported."
    model_name_dict = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229"
    }
    client = anthropic.Anthropic()
    
    class Model(BaseModel):
        def chat_complete(self, message):
            system_message, message = self._convert_to_anthropic_format(message)
            if system_message == "":
                system_message = anthropic.NOT_GIVEN
            response = client.messages.create(
                model=model_name_dict[name],
                system=system_message,
                messages=message,
                temperature=0,
                max_tokens=4096
            )
            if response.content:
                response_text = response.content[0]
                assert isinstance(response_text, anthropic.types.TextBlock)
                text = response_text.text
                if isinstance(text, str):
                    return text
            return ""

        @staticmethod
        def _convert_to_anthropic_format(message: list[dict[str, str]]):
            system_message = ""
            formatted_message = []
            for m in message:
                if m["role"] == "system":
                    system_message += m["content"]
                else:
                    formatted_message.append({
                        "role": m["role"],
                        "content": [{"type": "text", "text": m["content"]}]
                    })
            return system_message, formatted_message

    return Model()


class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
        """ The message should be a list of dictionaries with the following keys:
        - role: "system", "user", or "assistant" (all these roles should be supported)
        - content: the content of the message
        """
        pass

