import gradio as gr
import time
import re
from ..utils import ActionManager, form_prompt
from ..session import Session
CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
SYS_MSG_BASE = "You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step."
class LMGradioInterface:
    def __init__(self, session: Session, assist_session: Session = None, use_lm: bool=True, logger=None):
        self.session = session
        self.assist_session = assist_session
        self.logger = logger
        self.action_manager = ActionManager()
        self.is_busy = False
        self.use_lm = use_lm
        self.on_mount()

    def on_mount(self):
        if self.assist_session:
            title = f"{self.session.model_name}+{self.assist_session.model_name}"
        else:
            title = self.session.model_name
        with gr.Blocks(css=CSS, title=title) as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            infer_box = gr.Textbox("", interactive=False, label="Actions", max_lines=15)
            msg = gr.Textbox(placeholder="Type here...", label="Input")
            clear = gr.Button("Clear")
            use_lm = gr.Checkbox(label="Use LM framework", value=self.use_lm)
            def user(user_message, history):
                return "", history + [[user_message, None]], ""

            def bot(history, use_lm):
                converted_history = convert_history(history)
                if use_lm:
                    new_prompts = form_prompt(
                        converted_history[-1]["content"],
                        converted_history[:-1],
                        self.action_manager,
                        is_completed=True
                    )
                else:
                    new_prompts = [{"role": "system", "content": SYS_MSG_BASE},] + converted_history
                while self.is_busy:
                    time.sleep(0.1)
                self.is_busy = True
                history[-1][1] = ""
                if use_lm and self.assist_session:
                    self.assist_session.chat_complete(new_prompts)
                    streamer = self.assist_session.stream()
                else:
                    self.session.chat_complete(new_prompts)
                    streamer = self.session.stream()
                for next_text in streamer:
                    history[-1][1] += next_text
                    yield history
                self.is_busy = False


            def save_action(text):
                pattern = r"^action ([a-z]+).[ \n]*(.*)$"
                response = text
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

            def action_change(text, use_lm, infer_box):
                if not use_lm or self.is_busy:
                    yield infer_box
                    return
                actions, prompts = self.action_manager.read_action(
                    text, save_prompts=True
                )
                infer_box = "\n".join(actions)
                if prompts is None or len(actions) >= len(prompts):
                    yield infer_box
                    return

                self.is_busy = True
                new_prompts = form_prompt(
                    text,
                    convert_history(chatbot.value),
                    self.action_manager,
                    is_completed=False
                )
                self.session.chat_complete(new_prompts)
                response = ""
                if infer_box:
                    infer_box += "\n"
                for next_text in self.session.stream():
                    response += next_text
                    infer_box += next_text
                    yield infer_box
                yield infer_box
                if self.logger:
                    self.logger.info(response)
                self.is_busy = False
                save_action(response)


            msg.submit(user, [msg, chatbot], [msg, chatbot, infer_box], queue=False).then(
                bot, [chatbot, use_lm], chatbot
            )

            msg.change(action_change, [msg, use_lm, infer_box], [infer_box], show_progress=False)
            clear.click(lambda: None, None, chatbot, queue=False)
        self.demo = demo
    
    def run(self):
        demo = self.demo
        demo.queue()
        demo.launch()


def convert_history(history:list[list[str]]) -> list[dict]:
    result = []
    for dialog in history:
        if dialog[0]:
            result.append({"role": "user", "content": dialog[0]})
        if dialog[1]:
            result.append({"role": "assistant", "content": dialog[1]})
    return result
