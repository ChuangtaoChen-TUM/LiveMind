from typing import Callable
import random
import math
import nltk

DEFAULT_CHUNK_SIZE = 10

class BatchTextGenerator:
    def __init__(
        self,
        texts: list[str],
        strategy: str,
        delay_fn: Callable[[str], float],
        gen_config: dict = {}
    ):
        self.strategy = strategy
        self.gen_config = gen_config
        assert self.strategy in ["char", "word", "chunk", "token", "sentence"]
        # the delay function that takes next-to-be-generated text and returns the delay time
        self.delay_fn = delay_fn
        self.generators = []
        for text in texts:
            self.generators.append(TextGenerator(text, strategy, delay_fn, gen_config))
    
    def generate(self):
        if not self.generators:
            return None, 0
        next_texts = []
        delays = []
        for generator in self.generators:
            if generator.empty():
                next_texts.append(None)
                delays.append(0)
            else:
                next_text, delay = generator.generate()
                next_texts.append(next_text)
                delays.append(delay)
        return next_texts, delays

    def empty(self):
        return [generator.empty() for generator in self.generators]

class TextGenerator:
    def __init__(
        self,
        text: str,
        strategy: str,
        delay_fn: Callable[[str], float],
        gen_config: dict = {}
    ):
        self.strategy = strategy
        self.gen_config = gen_config
        assert self.strategy in ["char", "word", "chunk", "token", "sentence"]
        # the delay function that takes next-to-be-generated text and returns the delay time
        self.delay_fn = delay_fn
        self.text = getattr(self, f"gen_{self.strategy}")(text)

    def generate(self):
        if not self.text:
            return None, 0
        next_text = self.text[0]
        self.text = self.text[1:]
        delay = self.delay_fn(next_text)
        return next_text, delay

    def gen_char(self, text) -> str:
        return list(text)

    def gen_word(self, text):
        return nltk.word_tokenize(text)

    def gen_chunk(self, text):
        chunk_size = self.gen_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        text_in_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return text_in_chunks

    def gen_token(self, text):
        ...

    def gen_sentence(self, text):
        sentences = nltk.sent_tokenize(text)
        return sentences

    def empty(self):
        return not self.text

def build_delay_fn_char(mean, sigma):
    def delay_fn_char(text):
        text_len = len(text)
        total_mean = mean * text_len
        total_sigma = sigma * math.sqrt(text_len)
        delay = random.gauss(total_mean, total_sigma)
        return max(0, delay)
    return delay_fn_char

class GreedyStreamer:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        initial_text:str="",
        min_gen_len: int=1,
        append_space: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.text = initial_text
        assert min_gen_len > 0
        self.min_gen_len = min_gen_len
        self.append_space = append_space
    
    def stream(self, text: str) -> str:
        output_text = ""
        self.text += text
        tokenized_text = self.tokenizer(self.text)
        if len(tokenized_text) > self.min_gen_len:
            join_str = " " if self.append_space else ""
            output_text = join_str.join(tokenized_text[:self.min_gen_len])
            self.text = join_str.join(tokenized_text[self.min_gen_len:])
        else:
            output_text = ""
        return output_text

    def flush(self):
        output_text = self.text
        self.text = ""
        return output_text
