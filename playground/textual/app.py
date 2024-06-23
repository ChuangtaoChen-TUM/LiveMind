import re
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, TextArea
from textual.app import App
from typing import Callable, AsyncGenerator
from asyncio import create_task
from textual import log
from .component import MessageWrapper, ReactMessageWrapper, ReactChatMessage
from ..action_manager import ActionManager
from ..utils import form_prompt

class ChatbotApp(App):
    CSS_PATH = "./style/style.tcss"
    TITLE = "Chatbot"

    def __init__(
            self,
            model_name: str,
            assist_model_name: str,
            stream_method:Callable[[str], AsyncGenerator],
            assist_stream_method:Callable[[str], AsyncGenerator] = None,
            use_lm:bool = True,
            logger=None
        ):
        super().__init__()
        self.stream = stream_method
        self.assist_stream = assist_stream_method
        self.use_lm = use_lm
        self.chat_history = [
        ]
        self.action_manager = ActionManager()
        if assist_model_name:
            self.title = "Chatbot: " + model_name + "+" + assist_model_name
        else:
            self.title = "Chatbot: " + model_name
        self.running_task = None
        self.response = None
        self.logger = logger


    def compose(self):
        yield Header()
        yield Horizontal(
            Vertical(
                ScrollableContainer(id="history"),
                TextArea(id="input"),
                id="text_area_container",
            ),
            Vertical(
                Button(label="Exit", id="exit"),
                Button(label="Clear", id="clear"),
                Button(label="Send", id="send"),
                id="button_container",
            ),
            id="main_container",
        )
        yield Footer()


    async def on_mount(self):
        self.history_widget = self.query_one("#history", ScrollableContainer)
        for message in self.chat_history:
            self.history_widget.mount(MessageWrapper(message["content"], message["role"]))


    async def on_text_area_changed(self, event: TextArea.Changed):
        if event.text_area.id == "input":
            content = event.text_area.text
        else:
            return
        if not self.use_lm:
            return
        if self.running_task and not self.running_task.done():
            return

        await self.save_action()
        actions, prompts = self.action_manager.read_action(content, save_prompts=True)
        if prompts is None or len(actions) >= len(prompts):
            return

        new_prompts = form_prompt(
            content,
            self.chat_history,
            self.action_manager,
            is_completed=False
        )
        self.running_task = create_task(self.chat_complete(new_prompts))


    async def save_action(self):
        pattern = r"^action ([a-z]+).[ \n]*(.*)$"
        response = self.response
        self.response = None
        if response is None:
            return
        matched = re.match(pattern, response, re.DOTALL)
        if matched:
            if matched.group(1) not in ["background", "inference", "hypothesize", "wait"]:
                return
            if matched.group(1) == "wait":
                self.action_manager.write_action("")
            else:
                self.action_manager.write_action(response)
        else:
            return


    async def on_button_pressed(self, event: Button.Pressed):
        if self.running_task and not self.running_task.done():
            return
        if event.button.id == "send":
            input_widget = self.query_one("#input", TextArea)
            content = input_widget.text
            if content:
                await self.send_user_message(content)
        elif event.button.id == "clear":
            self.chat_history = []
            self.history_widget.remove_children()
        elif event.button.id == "exit":
            self.exit()


    async def send_user_message(self, content):
        input_widget = self.query_one("#input", TextArea)
        new_msg = {"role": "user", "content": content}
        if self.use_lm:
            new_prompts = form_prompt(
                content,
                self.chat_history,
                self.action_manager,
                is_completed=True
            )
        else:
            new_prompts = self.chat_history + [new_msg,]
        await self.send_message(new_msg)
        input_widget.text = ""
        if self.running_task and not self.running_task.done():
            await self.running_task
        self.running_task = create_task(self.get_response(new_prompts))


    async def send_message(self, message):
        content = message["content"]
        role = message["role"]
        new_msg = MessageWrapper(content, role)
        self.chat_history.append(message)
        self.history_widget.mount(new_msg)
        new_msg.scroll_visible()


    async def chat_complete(self, dialogs):
        if not self.stream:
            raise ValueError("Chatbot not set")
        response = self.stream(dialogs)
        new_content = ""
        async for each in response:
            new_content += each
        self.response = new_content
        log(new_content)
        if self.logger:
            self.logger.info(new_content)


    async def get_response(self, dialogs):
        if not self.stream:
            raise ValueError("Chatbot not set")
        if self.assist_stream:
            response = self.assist_stream(dialogs)
        else:
            response = self.stream(dialogs)
        response_widget = ReactMessageWrapper()
        message_widget = ReactChatMessage()
        message_widget.content = ""
        await self.history_widget.mount(response_widget)
        await response_widget.mount(message_widget)
        response_widget.scroll_visible()
        new_content = ""
        async for each in response:
            new_content += each
            message_widget.content = new_content
            message_widget.refresh()
            response_widget.scroll_visible()

        response_widget.remove()
        if self.logger:
            self.logger.info(new_content)
        new_msg = {"role": "assistant", "content": new_content}
        await self.send_message(new_msg)
