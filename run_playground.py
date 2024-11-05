import argparse
from playground import get_stream_model, SUPPORTED_MODELS
from playground.gradio import LMGradioInterface
from live_mind.formatter import LMFormat, LMFormatter, CoTFormatter
from live_mind import LMStreamController, CompleteStreamController
from live_mind.text import get_segmenter

FORMAT_MAP = {
    "u-pi"  : LMFormat.U_PI,
    "u-pli" : LMFormat.U_PLI,
    "ua-pil": LMFormat.UA_PIL,
    "u-spi" : LMFormat.U_SPI,
    "ua-spi": LMFormat.UA_SPI,
}
GRAUNLARITIES = ["char", "word", "sent", "clause"]
DEFAULT_MIN_LEN = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chat interface")
    parser.add_argument("-i",  "--infer-model",   metavar="M",    type=str, default="llama-3-70b", help=f"inference model, default: llama-3-70b, supported: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("-o",  "--out-model",     metavar="M",    type=str, default=None, help=f"output model, default: same as inference model, supported: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("-pf", "--prompt-format", metavar="FMT",  type=str, default="ua-spi", choices=FORMAT_MAP.keys(), help=f"prompt format, can be {', '.join(FORMAT_MAP.keys())}")
    parser.add_argument("-g",  "--granularity",   metavar="G",    type=str, default="clause", choices=GRAUNLARITIES, help=f"granularity of the text streamer, can be {', '.join(GRAUNLARITIES)}")
    parser.add_argument("--log",     action="store_false",  dest="log",  default=False, help="log the results")
    parser.add_argument("--min-len",             metavar="N", type=int, default=DEFAULT_MIN_LEN, help=f"minimum length of the segment if using sent or clause granularity, default: {DEFAULT_MIN_LEN}")

    args = parser.parse_args()

    infer_model_name = args.infer_model
    out_model_name = args.out_model

    if not infer_model_name:
        raise ValueError("Please specify the inference model")
    if not out_model_name:
        print("Warning: output model is not specified, using inference model as output model")
        out_model_name = infer_model_name
    infer_model = get_stream_model(infer_model_name)
    if infer_model_name != out_model_name:
        out_model = get_stream_model(out_model_name)
    else:
        out_model = infer_model


    prompt_format = FORMAT_MAP[args.prompt_format]
    formatter = LMFormatter(prompt_format)
    if args.granularity in ["sent", "clause"]:
        seg_kwargs = {"min_len": args.min_len}
    else:
        seg_kwargs = {}
    segmenter = get_segmenter(args.granularity, **seg_kwargs)
    lm_controller = LMStreamController(
        segmenter=segmenter,
        formatter=formatter,
        infer_model=infer_model,
        output_model=out_model,
    )
    base_controller = CompleteStreamController(CoTFormatter(), out_model)

    app = LMGradioInterface(lm_controller, base_controller)
    app.run()
