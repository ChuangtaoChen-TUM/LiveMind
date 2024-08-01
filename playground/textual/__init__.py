from .async_utils import async_gen
from .app import ChatbotApp
from ..session import Session

class LiveMindTextual:
    """ Chat interface using Textual interface """
    def __init__(
        self,
        session: Session,
        assist_session:Session|None=None,
        use_lm=True,
        logger=None
    ) -> None:
        self.session = session
        self.assist_session = assist_session
        model_name = self.session.model_name
        stream = self.stream
        assist_stream = self.assist_stream if assist_session else None
        if self.assist_session:
            assist_model_name = self.assist_session.model_name
        else:
            assist_model_name = None
        self.chat_app = ChatbotApp(
            model_name,
            assist_model_name,
            stream,
            assist_stream,
            use_lm,
            logger
        )

    def run(self):
        """ Run the chat interface """
        self.chat_app.run()

    async def stream(self, history):
        self.session.chat_complete(history)
        generator = self.session.stream()
        async for each_item in async_gen(generator):
            yield each_item

    async def assist_stream(self, history):
        self.assist_session.chat_complete(history)
        generator = self.assist_session.stream()
        async for each_item in async_gen(generator):
            yield each_item
