""" text streamers are classes to simulate the streaming input of text data
To use a text streamer, the texts to be streamed should be provided in advance.
"""
__all__ = ["TextStreamer"]

from collections.abc import Callable
from typing import Optional
from .abc import BaseTextStreamer

class TextStreamer(BaseTextStreamer):
    """ Simulator for typing of text data. The texts will be generated based on `current_time` in the simulation.
    - args:
        - text: the text to be streamed
        - delay_fn: a function that takes a text and returns the time required to type the text
        - granularity: how the text is typed. Default is "char". 
        - final_text: the final text attached to the end of the text stream. The final text does not have delay.
        - config: additional configuration for the generator
    """
    def __init__(
        self,
        text: str,
        delay_fn: Callable[[str], float],
        granularity: str = "char",
        final_text: Optional[str] = None,
        config: dict = {}, # additional configuration for the generator
    ):
        assert granularity in ["char", "chunk", "token"]
        self.delay_fn = delay_fn
        self.text = self.split(text, granularity, **config)
        self.final_text = final_text
        self.index = 0
        self.current_time = 0.0 # current time
        self.last_gen_time = 0.0 # when the last text was generated
        # 0 <= current_time - last_gen_time < delay_fn(next_text) always hold
        self.max_index = len(self.text)


    def next(self) -> str|None:
        """ Generate the next text according to the graunlarity.
            Current time will be updated to the time when the text is generated.
            Return `None` if the streamer reaches the end of the text, in this case current time will not be updated
        """
        if self.index >= self.max_index:
            return None

        next_text = self.text[self.index]
        delay = self.delay_fn(next_text)
        self.index += 1

        if self.index == self.max_index and self.final_text:
            next_text += self.final_text

        self.last_gen_time += delay
        self.current_time = self.last_gen_time
        return next_text


    def wait(self, delay: float) -> str|None:
        """ get text generated in a period of time, return `None` if no text is generated.
        No matter if any text is generated, the current time will be updated by the delay time.
        """
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
            self.index += 1
            delay -= next_time
            self.current_time += next_time
            self.last_gen_time = self.current_time

            if self.index == self.max_index and self.final_text:
                next_text += self.final_text
            texts.append(next_text)

        self.current_time += delay # add the remaining delay time to the current time

        if texts:
            return "".join(texts)
        return None


    def empty(self) -> bool:
        return self.index >= self.max_index


    def flush(self) -> str|None:
        """ Return the remaining text in the streamer, return `None` if no text is remaining.
        The current time will be updated to the time when the last text is generated.
        """
        if self.index >= self.max_index:
            return None

        remaining_texts = self.text[self.index:]
        remaining_text = "".join(remaining_texts)
        self.index = self.max_index
        delay = self.delay_fn(remaining_text)
        self.last_gen_time += delay
        self.current_time = self.last_gen_time

        if self.final_text:
            remaining_text += self.final_text
        return remaining_text


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
            case _:
                raise ValueError(f"granularity {granularity} is not supported.")
