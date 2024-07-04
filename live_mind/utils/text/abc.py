from abc import ABC, abstractmethod

class BaseTextStreamer(ABC):
    """ Abstract class for text streamer """
    @abstractmethod
    def next(self) -> str|None:
        """ Generate the next text and the delay time for the text """
        pass

    @abstractmethod
    def wait(self, delay: float) -> str|None:
        """ get text generated in a period of time, return `None` if no text is generated. Current time will be updated. """
        pass

    @abstractmethod
    def empty(self) -> bool:
        """ Check if the streamer is empty """
        pass
