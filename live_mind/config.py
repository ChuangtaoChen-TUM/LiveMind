""" Configuration file """
MMLU_PRO_PATH = "" # path to the MMLU-PRO dataset

# config the model path if you use 'get_model_vllm_example'
LLAMA_3_8B_PATH = "" # replace this with your own path
LLAMA_3_70B_PATH = "" # replace this with your own path

"""
The get model function should return a model with the following methods:
- chat_complete: takes a batched list of messages and returns a list of responses
similar to `chat.completions.create` in OpenAI's API but its in batch.
for example:
model = get_model("llama-3-8b")
messages = [
    [
        {
            "role": "user",
            "content": "This is message 1."
        },
    ],
    [
        {
            "role": "system",
            "content": "This is a system message."
        },
        {
            "role": "user",
            "content": "This is message 2."
        }
    ]
]
responses: list = model.chat_complete(messages)

The responses should be a list of responses corresponding to the input messages.
each response should have the following format: (similar to OpenAI's API)
{
    "choices": [
        {
            "message": {
                "content": "This is a response message."
            }
        }
    ]
    "usage": {
        "completion_tokens": 1,
        "prompt_tokens": 2,
        "total_tokens": 3
    }
}"""

def get_model(name: str):
    assert name in ["llama-3-8b", "llama-3-70b"]
    raise NotImplementedError("Please implement your own model function.")
    return get_model_vllm_example(name) # You can also use this example implementation

def get_model_vllm_example(name: str):
    """ This is an example implementation of the model function using VLLM. You can replace this with your own implementation."""
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
        raise ValueError(f"Please replace the model path for {name} at live_mind/default/__init__.py.")
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
        def chat_complete(self, messages):
            prompts = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer.batch_encode_plus(
                prompts
            )["input_ids"]
            prompts_len = [len(input) for input in inputs]
            vllm_results = model.generate(prompt_token_ids=inputs, sampling_params=param, use_tqdm=False)
            results: list[list[int]] = []
            for vllm_result in vllm_results:
                results.append(vllm_result.prompt_token_ids+vllm_result.outputs[0].token_ids)
            total_len = [len(result) for result in results]
            gen_len = [total_len[i] - prompts_len[i] for i in range(len(total_len))]
            # remove the input tokens from the output
            for i in range(len(results)):
                assert len(results[i]) >= len(inputs[i])
                assert results[i][:len(inputs[i])] == inputs[i]
                results[i] = results[i][len(inputs[i]):]
                if results[i][-1] in stop_token_ids:
                    results[i] = results[i][:-1]

            generated_texts = tokenizer.batch_decode(
                results,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False
            )
            outputs = [
                {
                    "choices": [{"message": {"content": generated_texts[i], "role": "assistant"}}],
                    "usage": {
                        "completion_tokens": gen_len[i],
                        "prompt_tokens": prompts_len[i],
                        "total_tokens": total_len[i]
                    }
                } 
                for i in range(len(generated_texts))
            ]

            return outputs
    model_with_chat_complete = Model()
    return model_with_chat_complete
