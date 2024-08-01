from textual.containers import Container
from textual.widgets import Static
from textual.reactive import reactive

class MessageWrapper(Container):
    def __init__(self, content: str, role: str):
        super().__init__()
        self.content = content
        self.role = role
        self.classes = role+"-wrapper"

    def compose(self):
        yield ChatMessage(self.content, self.role)


class ChatMessage(Static):
    def __init__(self, content: str, role: str):
        super().__init__()
        if role == "system":
            self.content = f"System: {content}"
        else:
            self.content = content
        self.role = role
        self.classes = role+"-message"

    def render(self):
        return self.content


class ReactMessageWrapper(Container):
    def __init__(self):
        super().__init__()
        self.classes = "assistant-wrapper"


class ReactChatMessage(Static):
    content = reactive("", layout=True)
    def __init__(self):
        super().__init__()
        self.classes = "assistant-message"

    def render(self):
        return self.content
