""" text streamers are classes to simulate the streaming input of text data
To use a text streamer, the texts to be streamed should be provided in advance.
"""
__all__ = ["TextStreamer"]

from collections.abc import Callable
from typing import Optional
from .segmenter import nltk_sent_segmenter
from .abc import BaseTextStreamer

class TextStreamer(BaseTextStreamer):
    """ Streamer for text data. The texts will be generated based on `current_time` in the simulation.
    
    `next`: generate next text and current time will be updated to the time when the text is generated.
    
    `wait`: get text generated in a period of time. Current time will be updated based on `delay`.

    """
    def __init__(
        self,
        text: str,
        granularity: str,
        delay_fn: Callable[[str], float],
        final_text: Optional[str] = None,
        config: dict = {}, # additional configuration for the generator
    ):
        assert granularity in ["char", "chunk", "token", "sentence"]
        self.delay_fn = delay_fn
        self.text = self.split(text, granularity, **config)
        if final_text:
            self.text.append(final_text)
        self.index = 0
        self.current_time = 0.0 # current time
        self.last_gen_time = 0.0 # when the last text was generated
        # 0 <= current_time - last_gen_time < delay_fn(next_text) always hold
        self.max_index = len(self.text)


    def next(self) -> str|None:
        """ Generate the next text. Current time will be updated to the time when the text is generated.
        Return `None` if the streamer reaches the end of the text, in this case current time will not be updated """
        if self.index >= self.max_index:
            return None
        next_text = self.text[self.index]
        self.index += 1
        delay = self.delay_fn(next_text)
        self.last_gen_time += delay
        self.current_time = self.last_gen_time
        return next_text


    def wait(self, delay: float) -> str|None:
        """ get text generated in a period of time, return `None` if no text is generated. Current time will be updated. """
        assert delay >= 0
        texts = []
        while True:
            # if the streamer reaches the end of the text
            if self.index >= self.max_index:
                break
            next_text = self.text[self.index]
            # time required to generate the next text
            next_time = self.last_gen_time - self.current_time + self.delay_fn(next_text)
            if delay < next_time:
                # time is not enough to generate the next text
                break
            # update delay, current time and last generation time
            delay -= next_time
            self.current_time += next_time
            self.last_gen_time = self.current_time
            texts.append(next_text)
            self.index += 1
        self.current_time += delay # add the remaining delay time to the current time

        if texts:
            return "".join(texts)
        return None


    def empty(self) -> bool:
        return self.index >= self.max_index


    @staticmethod
    def split(text: str, granularity: str, **kwargs) -> list[str]:
        match granularity:
            case "char":
                return list(text)
            case "chunk":
                try:
                    chunk_size = kwargs["chunk_size"]
                except KeyError:
                    raise ValueError("set config['chunk_size'] to specify the chunk size for the chunk granularity.")
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            case "token":
                raise NotImplementedError("token granularity is not implemented yet.")
            case "sentence":
                return nltk_sent_segmenter(text)
            case _:
                raise ValueError(f"granularity {granularity} is not supported.")
