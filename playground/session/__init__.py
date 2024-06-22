from transformers import TextIteratorStreamer, AutoTokenizer
from queue import Queue
from threading import Thread
from ..iter_vllm import ItervLLM
from ..config import LLAMA_3_70B_PATH, LLAMA_3_8B_PATH
from .generate import generate_thread_wrapper

EXTRA_TER = "<|eot_id|>"

class Session:
    """ Session class """
    def __init__(
        self,
        model: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int = 4096,
    ):
        assert model in ["llama-3-8b", "llama-3-70b"]
        if model == "llama-3-8b":
            model_path = LLAMA_3_8B_PATH
        elif model == "llama-3-70b":
            model_path = LLAMA_3_70B_PATH
        if model_path == "":
            raise ValueError(f"Please specify the path to the model {model}")
        self.model_name = model
        self.generator = ItervLLM(
            model_path, model_path, tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
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

    def chat_complete(self, dialog):
        prompt = self.tokenizer.apply_chat_template(
            [dialog,],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer.batch_encode_plus(
            prompt
        )
        self.task_queue.put(inputs)
        return
    
    def stream(self):
        for next_text in self.streamer:
            if next_text.endswith(EXTRA_TER):
                break
            yield next_text

    def flush(self):
        response = ""
        for next_text in self.stream():
            response += next_text
        return response
