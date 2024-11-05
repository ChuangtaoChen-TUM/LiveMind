from transformers import TextIteratorStreamer, AutoTokenizer
from queue import Queue
from typing import Generator
from threading import Thread
from live_mind.abc import BaseStreamModel
from config import LLAMA_3_70B_PATH, LLAMA_3_8B_PATH
from .iter_vllm import ItervLLM

EXTRA_TER = "<|eot_id|>"

# Model path and GPU memory utilization
# if memory is not enough, consider increasing the utilization
MODEL_DICT = {
    "llama-3-8b": (LLAMA_3_8B_PATH, 0.3),
    "llama-3-70b": (LLAMA_3_70B_PATH, 0.6),
}

def generate_thread_wrapper(generator: ItervLLM, task_queue: Queue, gen_config: dict):
    """ Wrapper for the generation thread """
    while True:
        inputs = task_queue.get()
        if inputs is None:
            break
        generator.iter_generate(
            **inputs,
            **gen_config
        )

class Session(BaseStreamModel):
    """ Session class """
    def __init__(
        self,
        model: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int = 8192,
    ):
        assert model in ["llama-3-8b", "llama-3-70b"]
        model_path, gpu_memory_utilization = MODEL_DICT[model]
        if model_path == "":
            raise ValueError(f"Please specify the path to the model {model} in config.py")
        self.model_name = model
        self.generator = ItervLLM(
            model_path, model_path, tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True,
        )
        self.terminators = [self.tokenizer.eos_token_id, EXTRA_TER]
        self.task_queue = Queue()
        gen_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_gen_len,
            "eos_token_id": [self.tokenizer.eos_token_id, EXTRA_TER],
            "streamer": self.streamer,
        }
        self.generate_thread = Thread(
            target=generate_thread_wrapper,
            args=(self.generator, self.task_queue, gen_config),
            daemon=True
        )
        self.generate_thread.start()

    def chat_complete(self, message: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [message,],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer.batch_encode_plus(
            prompt
        )
        self.task_queue.put(inputs)
        response = ""
        for next_text in self.streamer:
            response += next_text
        return response

    def stream(self, message: list[dict[str, str]]) -> Generator[str, None, None]:
        prompt = self.tokenizer.apply_chat_template(
            [message,],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer.batch_encode_plus(
            prompt
        )
        self.task_queue.put(inputs)
        for next_text in self.streamer:
            yield next_text
