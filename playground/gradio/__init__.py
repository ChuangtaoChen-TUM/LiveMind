__all__ = ["LMGradioInterface"]

import gradio as gr
from threading import Lock
from live_mind.controller.abc import BaseStreamController
CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
class LMGradioInterface:
    def __init__(
        self,
        lm_controller: BaseStreamController,
        base_controller: BaseStreamController
    ):
        self.lm_controller = lm_controller
        self.base_controller = base_controller
        self.lock = Lock()
        self.use_lm = True
        self.infer_msg = ""
        self.input_msg = ""
        self.on_mount()

    def on_mount(self):
        title = "Live Mind Chat Interface"
        with gr.Blocks(css=CSS, title=title) as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            infer_box = gr.Textbox("", interactive=False, label="Actions", max_lines=15)
            msg = gr.Textbox(placeholder="Type here...", label="Input")
            clear = gr.Button("Clear")
            use_lm = gr.Checkbox(label="Use LM framework", value=self.use_lm)
            show_infer = gr.Checkbox(label="Show inference", value=True)

            def clear_input(msg):
                self.input_msg += msg
                return ""

            def update_input():
                self.input_msg += "\n"

            def action_submit(use_lm, chatbot):
                chatbot += [[self.input_msg, ""]]
                with self.lock:
                    controller = self.lm_controller if use_lm else self.base_controller
                    for response in controller.iter_call(self.input_msg, stream_end=True):
                        for text in response:
                            chatbot[-1][1] += text
                            yield chatbot
                    yield chatbot

            def action_change(text, use_lm):
                if not use_lm:
                    return

                with self.lock:
                    text = self.input_msg + text
                    for response in self.lm_controller.iter_call(text):
                        if self.infer_msg != "":
                            self.infer_msg += "\n"
                        for s in response:
                            self.infer_msg += s
                            yield self.infer_msg
                    yield self.infer_msg

            def action_clear():
                self.infer_msg = ""
                self.input_msg = ""
                self.base_controller.reset()
                self.lm_controller.reset()
                return None, self.infer_msg

            def change_visibility(show_infer):
                value = show_infer
                if value:
                    return gr.Textbox(visible=True)
                else:
                    return gr.Textbox(visible=False)

            msg.submit(clear_input, [msg], msg, queue=True).then(action_submit, [use_lm, chatbot], [chatbot], queue=True).then(update_input, [], queue=True)
            msg.change(action_change, [msg, use_lm], infer_box, show_progress=False, queue=True)
            clear.click(action_clear, [], [chatbot, infer_box], queue=True)
            show_infer.change(change_visibility, show_infer, infer_box, show_progress=False)
        self.demo = demo


    def run(self):
        demo = self.demo
        demo.launch()
