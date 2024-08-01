import argparse
import logging
from playground.session import Session
from playground.textual import LiveMindTextual
from playground.gradio import LMGradioInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chat interface")
    parser.add_argument("--textual", action="store_true", help="Run the textual interface")
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio interface")
    parser.add_argument("--model", type=str, help="inference model")
    parser.add_argument("--assist_model", type=str, help="assist model")
    parser.add_argument("--use_lm", action="store_true", help="Use LiveMind framework")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    args = parser.parse_args()

    model_name = args.model
    assist_model_name = args.assist_model
    if assist_model_name == model_name:
        raise ValueError("Model and assist model cannot be the same")
    if not args.use_lm and assist_model_name:
        print("Warning: assist model will not be used without LiveMind framework")
        assist_model_name = None
    if args.textual and args.gradio:
        raise ValueError("Only one interface can be selected")
    if args.log:
        logger: logging.Logger|None = logging.getLogger('playground_logger')
        assert isinstance(logger, logging.Logger)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('./playground/log.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger = None
    if args.textual:
        session = Session(model_name)
        assist_session = Session(assist_model_name) if assist_model_name else None
        textual_interface = LiveMindTextual(session, assist_session, args.use_lm, logger)
        textual_interface.run()
    if args.gradio:
        session = Session(model_name)
        assist_session = Session(assist_model_name) if assist_model_name else None
        gradio_interface = LMGradioInterface(session, assist_session, args.use_lm, logger)
        gradio_interface.run()
