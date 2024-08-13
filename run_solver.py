""" This script is used to run the solver for real-time latency measure on the MMLU Pro dataset. """
import json
import logging
import pathlib
import argparse
import time
from tqdm import tqdm
from live_mind import LMController, CompleteCoTController, BaseController
from live_mind.format.formatter import LMFormatter, CoTFormatter, LMFormat
from live_mind.text import (
    TextStreamer,
    get_segmenter,
)
from live_mind.utils.dataset import GSM8kDataset, MMLUProDataset, BaseDataset, MMLUDataset
from config import BaseModel, MMLU_PRO_PATH, GSM8K_PATH, MMLU_PATH, get_model


def main(
    controller: BaseController,
    inference_model: BaseModel,
    output_model: BaseModel,
    dataset: BaseDataset,
    input_speed: int,
    output_file: str|None=None,
):
    """ Run the solver on the MMLU Pro dataset and log the results if needed. """
    # warm up the models
    inference_model.chat_complete([{"role": "user", "content": "Hello"}])
    output_model.chat_complete([{"role": "user", "content": "Hello"}])
    entry_list = []
    num_correct = 0
    num_total = 0


    def delay_fn(text: str) -> float: # delay function to simulate the typing speed (seconds)
        return len(text) / input_speed * 60

    tqdm_bar = tqdm(dataset.selected_questions)
    for entry in tqdm_bar:
        question = entry["question"]
        final_text = dataset.add_str(entry)
        streamer = TextStreamer(
            question,
            delay_fn=delay_fn,
            final_text=final_text
        )
        # Initialize the information to be recorded
        total_gen_time: float = 0  # the total time for the model to generate the responses
        # actions consists of the new prompt and actions taken by the model at each step
        # each list in the `actions` list is composed of the new prompt (the first item) and the actions taken by the model (the rest of the items)
        actions: list[list[str]] = []

        # current input text
        input_text = ""
        gen_time = 0.0
        new_prompt = ""
        while True:
            # the streamers is waiting for the LLM's response
            next_text = streamer.wait(gen_time)
            if next_text is None:
                # no more text during LLM's response, wait until the next text arrives
                next_text = streamer.next()
                if next_text is None:
                    # the streamer reaches the end (this line is only for type hint)
                    break
            input_text += next_text
            stream_end = streamer.empty() # This is the last text
            resp_gen = controller(input_text, stream_end=stream_end)

            new_prompt += next_text
            gen_time = 0.0
            step_actions = []
            start_time = time.time()
            response = ""
            try:
                for response in resp_gen:
                    step_gen_time = time.time() - start_time
                    step_actions.append(response)
                    gen_time += step_gen_time
                    start_time = time.time()
            except ValueError:
                streamer.flush()
                stream_end = True
                total_gen_time += gen_time
                gen_time =  0.0

            if step_actions:
                actions.append([new_prompt]+step_actions)
                new_prompt = ""

            total_gen_time += gen_time

            if stream_end: # the stream ends
                # overhead_time is the inference model is still generating response when
                # the final text is ready (causing additional latency)
                overhead_time = streamer.current_time - streamer.last_gen_time
                streamer.wait(gen_time) # update current time to the streamer
                controller.reset()
                break

        # latency is defined as the time between the last text is generated and the current time
        latency = streamer.current_time - streamer.last_gen_time
        if new_prompt:
            actions.append([new_prompt])

        is_correct: bool = dataset.verify_answer(response, entry["answer"])
        if is_correct:
            num_correct += 1
        num_total += 1

        time_info = {
            "latency": latency,
            "total_gen_time": total_gen_time,
            "overhead_time": overhead_time
        }

        # save info to the entry
        entry["actions"] = actions
        entry["time_info"] = time_info
        entry["correct"] = is_correct
        entry_list.append(entry)

        # verify the answer
        tqdm_bar.set_description(f"Accuracy: {num_correct/num_total:.2f}")

    print(f"Accuracy: {num_correct/num_total:.2f}")
    if output_file:
        print(f"Writing results to {output_file}")
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(entry_list, file, indent=4)

FORMAT_MAP = {
    "u-pi"  : LMFormat.U_PI,
    "u-pli" : LMFormat.U_PLI,
    # "u-pil" : LMFormat.U_PIL,
    # "u-ip"  : LMFormat.U_IP,
    # "u-ipl" : LMFormat.U_IPL,
    "ua-pil": LMFormat.UA_PIL,
    "u-spi" : LMFormat.U_SPI,
    "ua-spi": LMFormat.UA_SPI,
    # "ua-pf" : LMFormat.UA_PF,
}
GRAUNLARITIES = ["char", "word", "sent", "clause"]
DATASET_MAP = {
    "mmlu-pro": (MMLUProDataset, MMLU_PRO_PATH),
    "gsm8k": (GSM8kDataset, GSM8K_PATH),
    "mmlu": (MMLUDataset, MMLU_PATH),
}

DEFAULT_NUM_QUESTIONS = 1024
DEFAULT_SUM_LEN = -1
DEFAULT_MIN_LEN = 10
DEFAULT_INPUT_SPEED = 240 # characters per minute
SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--infer-model",   metavar="M",    type=str, default=None, help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("-o",  "--out-model",     metavar="M",    type=str, default=None, help="output model: llama-3-8b, llama-3-70b")
    parser.add_argument("-pf", "--prompt-format", metavar="FMT",  type=str, default=None, choices=FORMAT_MAP.keys(), help=f"prompt format, can be {', '.join(FORMAT_MAP.keys())}")
    parser.add_argument("-g",  "--granularity",   metavar="G",    type=str, default=None, choices=GRAUNLARITIES, help=f"granularity of the text streamer, can be {', '.join(GRAUNLARITIES)}")
    parser.add_argument("-d",  "--dataset",       metavar="D",    type=str, default=None, choices=DATASET_MAP.keys(), help=f"dataset to run the solver on, can be {', '.join(DATASET_MAP.keys())}")
    parser.add_argument("-f",  "--output-file",   metavar="File", type=str, nargs="?", default=False, const=True, help="output results to a json file")
    parser.add_argument("-is", "--input-speed",   metavar="S",    type=int, default=DEFAULT_INPUT_SPEED, help=f"input speed in characters per minute, default: {DEFAULT_INPUT_SPEED}")
    parser.add_argument("--no-lm",   action="store_false",  dest="lm",   default=True,  help="disable LiveMind framework, use baseline solver instead")
    parser.add_argument("--no-wait", action="store_false",  dest="wait", default=True,  help="disable waiting for the model to generate the response")
    parser.add_argument("--log",     action="store_false",  dest="log",  default=False, help="log the results")
    parser.add_argument("-n", "--num-questions", metavar="N", type=int, default=DEFAULT_NUM_QUESTIONS, help=f"number of questions per category, -1 for all questions, default: {DEFAULT_NUM_QUESTIONS}")
    parser.add_argument("--min-len",             metavar="N", type=int, default=DEFAULT_MIN_LEN, help=f"minimum length of the segment if using sent or clause granularity, default: {DEFAULT_MIN_LEN}")
    # parser.add_argument("--sum-len",             metavar="N", type=int, default=DEFAULT_SUM_LEN,       help=f"summarization length, default: {DEFAULT_SUM_LEN}")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the output file if it exists")
    args = parser.parse_args()

    # check arguments
    # load the dataset
    dataset_name: str = args.dataset
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Invalid dataset, please choose from {', '.join(DATASET_MAP.keys())}")
    dataset_class, dataset_path = DATASET_MAP[dataset_name]

    if not dataset_path:
        raise ValueError(f"Please set the path to the {dataset_name} dataset in config.py")

    num_questions = int(args.num_questions)
    assert num_questions == -1 or num_questions > 0

    # load the dataset
    dataset = dataset_class(dataset_path)
    dataset.select(num_questions, randomize=True, seed=SEED, split='test')

    # check framework settings
    use_lm: bool = args.lm
    infer_model_name: str|None = args.infer_model
    out_model_name: str|None = args.out_model

    if use_lm:
        if infer_model_name is None:
            raise ValueError("Please specify the inference model")
        if args.prompt_format is None:
            raise ValueError("Please specify the prompt format for the LiveMind framework")
        if args.granularity is None:
            raise ValueError("Please specify the granularity for the LiveMind framework")
        if out_model_name is None:
            print("Warning: --out-model is not set, use the inference model as output model")
            out_model_name = infer_model_name
        if args.granularity not in ["sent", "clause"] and args.min_len != DEFAULT_MIN_LEN:
            print("Warning: --granularity is not 'sent' or 'clause', the minimum length will be ignored")
        if args.wait is not True and args.granularity not in ["sent", "clause"]:
            print("Warning: granularity is not 'sent' or 'clause', the waiting setting will be ignored")
            args.wait = True
    else: # baseline
        if out_model_name is None:
            raise ValueError("Please specify the output model")
        if infer_model_name is not None:
            print("Warning: --no-lm is set, the inference model will be ignored")
        if args.prompt_format is not None:
            print("Warning: --no-lm is set, the prompt format will be ignored")
        if args.granularity is not None:
            print("Warning: --no-lm is set, the granularity will be ignored")
        if args.min_len != DEFAULT_MIN_LEN:
            print("Warning: --no-lm is set, the minimum length will be ignored")
        if args.wait is not True:
            print("Warning: --no-lm is set, the waiting setting will be ignored")
        # if args.sum_len != DEFAULT_SUM_LEN:
        #     print("Warning: --no-lm is set, the summarization length will be ignored")

    # logger configuration
    logger: logging.Logger|None = None
    if args.log:
        # Configure logging
        logger = logging.getLogger("lm_solver")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("./output/log.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # set the solver
    if use_lm: # LiveMind framework
        assert infer_model_name
        inference_model = get_model(infer_model_name)
        if out_model_name != infer_model_name:
            output_model = get_model(out_model_name)
        else:
            output_model = inference_model
        if args.granularity in ["sent", "clause"]:
            seg_kwargs = {"min_len": args.min_len}
        else:
            seg_kwargs = {}
        segmenter = get_segmenter(args.granularity, **seg_kwargs)
        format = FORMAT_MAP[args.prompt_format]
        formmatter = LMFormatter(format, args.wait)

        if format == LMFormat.UA_PF:
            if not inference_model.support_prefill or not output_model.support_prefill:
                raise ValueError("The model does not support prefilling, please choose a different prompt format")

        controller: BaseController = LMController(
            segmenter,
            formmatter,
            inference_model,
            output_model,
            # summarize_len=args.sum_len,
            answer_format=dataset.answer_format,
            logger=logger
        )
    else: # baseline
        output_model = get_model(out_model_name)
        inference_model = output_model
        controller = CompleteCoTController(
            CoTFormatter(),
            output_model=output_model,
            answer_format=dataset.answer_format,
        )

    # set the output file
    if isinstance(args.output_file, bool):
        if args.output_file: # output file set but not given
            if use_lm:
                g_str = args.granularity
                # sum_str = f"sum-{args.sum_len}" if args.sum_len != DEFAULT_SUM_LEN else "no-sum"
                num_q_str = f"{args.num_questions}q"
                input_speed_str = f"{args.input_speed}cpm"
                if args.wait is not True:
                    wait_str = "no-wait"
                else:
                    wait_str = "wait"
                output_file: str|None = f"./output/{dataset_name}/lm_{infer_model_name}_{out_model_name}_{args.prompt_format}_{g_str}_{input_speed_str}_{wait_str}_{num_q_str}.json"
            else:
                cot_str = "cot"
                output_file = f"./output/{dataset_name}/base_{out_model_name}_{cot_str}_{args.num_questions}q.json"
            print(f"-f is set but file name is not given, output to '{output_file}'")
        else:
            output_file = None

    if output_file is not None and not output_file.endswith(".json"):
        raise ValueError("Output file must be a json file")

    # if file exists, exit
    if output_file and pathlib.Path(output_file).exists():
        if args.overwrite:
            print(f"Output file {output_file} already exists, overwrite it")
        else:
            raise FileExistsError(f"Output file {output_file} already exists, please use --overwrite to overwrite it")
    main(
        controller=controller,
        inference_model=inference_model,
        output_model=output_model,
        input_speed=args.input_speed,
        dataset=dataset,
        output_file=output_file,
    )
