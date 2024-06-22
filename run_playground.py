import argparse
from playground.session import Session
from playground.textual import LiveMindTextual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chat interface")
    parser.add_argument("--textual", action="store_true", help="Run the textual interface")
    parser.add_argument("--model", type=str, help="inference model")
    parser.add_argument("--assist-model", type=str, help="assist model")
    parser.add_argument("--use_lm", action="store_true", help="Use LiveMind framework")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    args = parser.parse_args()

    model_name = args.model
    assist_model_name = args.assist_model
    if not args.use_lm and assist_model_name:
        print("Warning: assist model will not be used without LiveMind framework")
    if args.textual:
        session = Session(model_name)
        assist_session = Session(assist_model_name) if assist_model_name else None
        textual_interface = LiveMindTextual(session, assist_session, args.use_lm, args.log)
        textual_interface.run()
