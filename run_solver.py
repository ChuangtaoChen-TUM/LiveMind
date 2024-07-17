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
from live_mind.utils.dataset import GSM8kDataset, MMLUProDataset, BaseDataset
from config import BaseModel, MMLU_PRO_PATH, GSM8K_PATH, get_model


def main(
    controller: BaseController,
    inference_model: BaseModel,
    output_model: BaseModel,
    dataset: BaseDataset,
    output_file: str|None=None,
):
    """ Run the solver on the MMLU Pro dataset and log the results if needed. """
    # warm up the models
    inference_model.chat_complete([{"role": "user", "content": "Hello"}])
    output_model.chat_complete([{"role": "user", "content": "Hello"}])
    entry_list = []
    num_correct = 0
    num_total = 0
    def delay_fn(text: str): # delay function to simulate the typing speed
        return 0.25*len(text)

    tqdm_bar = tqdm(dataset.selected_questions)
    for entry in tqdm_bar:
        question = entry["question"]
        final_text = dataset.add_str(entry)
        streamer = TextStreamer(
            question,
            granularity="char", # typing character by character
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
            for response in resp_gen:
                step_gen_time = time.time() - start_time
                step_actions.append(response)
                gen_time += step_gen_time
                start_time = time.time()

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
    "u-pil" : LMFormat.U_PIL,
    "u-ip"  : LMFormat.U_IP,
    "u-ipl" : LMFormat.U_IPL,
    "ua-pil": LMFormat.UA_PIL,
    "u-spi" : LMFormat.U_SPI,
    "ua-spi": LMFormat.UA_SPI
}

DEFAULT_NUM_QUESTIONS = 1000
DEFAULT_CHUNK_SIZE = 20
DEFAULT_SUM_LEN = 5
DEFAULT_MIN_LEN = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--infer-model",   metavar="M",    type=str, default=None, help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("-o",  "--out-model",     metavar="M",    type=str, default=None, help="output model: llama-3-8b, llama-3-70b")
    parser.add_argument("-pf", "--prompt-format", metavar="FMT",  type=str, default=None, help=f"prompt format, can be {', '.join(FORMAT_MAP.keys())}")
    parser.add_argument("-g",  "--granularity",   metavar="G",    type=str, default=None, help=f"granularity of the text streamer, can be {', '.join(['char', 'chunk', 'sent', 'clause'])}")
    parser.add_argument("-d",  "--dataset",       metavar="D",    type=str, default=None, help="dataset to run the solver on, can be mmlu_pro or gsm8k")
    parser.add_argument("-f",  "--output-file",   metavar="File", type=str, nargs="?", default=False, const=True, help="output results to a json file")
    parser.add_argument("--no-hypo", action="store_false",  dest="hypo", default=True, help="disable hypothesize step after wait")
    parser.add_argument("--no-sum",  action="store_false",  dest="sum",  default=True, help="disable summarization step")
    parser.add_argument("--no-lm",   action="store_false",  dest="lm",   default=True, help="disable LiveMind framework, use baseline solver instead")
    parser.add_argument("--no-cot",  action="store_false",  dest="cot",  default=True, help="disable chain-of-thoughts when using baseline solver")
    parser.add_argument("--log",     action="store_false",  dest="log",  default=True, help="log the results")
    parser.add_argument("--no-retrieve_all", action="store_true", dest="retrieve_all", default=True, help="only retrieve one next prompt instead of all available prompts at each step")
    parser.add_argument("-n", "--num-questions", metavar="N", type=int, default=DEFAULT_NUM_QUESTIONS, help=f"number of questions per category, default: {DEFAULT_NUM_QUESTIONS}")
    parser.add_argument("--chunk-size",          metavar="N", type=int, default=DEFAULT_CHUNK_SIZE,    help=f"chunk size if using chunk granularity, default: {DEFAULT_CHUNK_SIZE}")
    parser.add_argument("--min-len",             metavar="N", type=int, default=DEFAULT_MIN_LEN, help=f"minimum length of the segment if using sent or clause granularity, default: {DEFAULT_MIN_LEN}")
    parser.add_argument("--sum-len",             metavar="N", type=int, default=DEFAULT_SUM_LEN,       help=f"summarization length, default: {DEFAULT_SUM_LEN}")
    args = parser.parse_args()

    assert args.num_questions == -1 or args.num_questions > 0

    logger: logging.Logger|None = None
    if args.log:
        # Configure logging
        logger = logging.getLogger("lm_solver")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("./output/log.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.dataset == "mmlu_pro":
        if not MMLU_PRO_PATH:
            raise ValueError("Please set the path to the MMLU-Pro dataset in config.py")
        dataset: BaseDataset = MMLUProDataset(MMLU_PRO_PATH)
    elif args.dataset == "gsm8k":
        if not GSM8K_PATH:
            raise ValueError("Please set the path to the GSM8K dataset in config.py")
        dataset = GSM8kDataset(GSM8K_PATH)
    else:
        raise ValueError("Invalid dataset, please choose either 'mmlu_pro' or 'gsm8k'")

    if not args.lm: # baseline
        if args.out_model is None:
            raise ValueError("Please specify the output model")
        if args.infer_model is None:
            print("Warning: --no-lm is set, the inference model will be ignored")
        if args.prompt_format is not None:
            print("Warning: --no-lm is set, the prompt format will be ignored")
        if args.granularity is not None:
            print("Warning: --no-lm is set, the granularity will be ignored")
        if args.chunk_size != DEFAULT_CHUNK_SIZE:
            print("Warning: --no-lm is set, the chunk size will be ignored")
        if args.min_len != DEFAULT_MIN_LEN:
            print("Warning: --no-lm is set, the minimum length will be ignored")
        if not args.hypo:
            print("Warning: --no-lm is set, the hypothesis is already disabled")
        if not args.sum:
            print("Warning: --no-lm is set, the summarization is already disabled")
        if not args.retrieve_all:
            print("Warning: --no-lm is set, the retrieve-all is already disabled")
        if args.sum_len != DEFAULT_SUM_LEN:
            print("Warning: --no-lm is set, the summarization length will be ignored")
        output_model = get_model(args.out_model)
        if output_model is None:
            raise ValueError("Invalid output model")
        inference_model = output_model
        controller: BaseController = CompleteCoTController(
            CoTFormatter(args.cot),
            output_model=output_model
        )
    else:
        if args.infer_model is None:
            raise ValueError("Please specify the inference model")
        if args.out_model is None:
            print("Warning: --out-model is not set, use the inference model as the output model")
        if args.chunk_size != DEFAULT_CHUNK_SIZE and args.granularity != "chunk":
            print("Warning: --chunk-size is set, but the granularity is not 'chunk', the chunk size will be ignored")
        if not args.sum and args.sum_len != DEFAULT_SUM_LEN:
            print("Warning: --use-sum is not set, the summarization length will be ignored")
        if not args.cot:
            print("Warning: --no-cot is set, when using LiveMind framework, the chain-of-thoughts setting will be ignored")
        inference_model = get_model(args.infer_model)
        if not inference_model:
            raise ValueError("Invalid inference model")
        if args.out_model:
            output_model = get_model(args.out_model)
        else:
            output_model = inference_model
            args.out_model = args.infer_model
        if args.prompt_format is None:
            raise ValueError("Please specify the prompt format")
        if args.granularity is None:
            raise ValueError("Please specify the granularity")
        seg_kwargs = {}
        if args.granularity == "chunk":
            seg_kwargs["chunk_size"] = args.chunk_size
        elif args.chunk_size != DEFAULT_CHUNK_SIZE:
            print("Warning: --granularity is not 'chunk', the chunk size will be ignored")
    
        if args.granularity == "sent" or args.granularity == "clause":
            seg_kwargs["min_len"] = args.min_len
        elif args.min_len != DEFAULT_MIN_LEN:
            print("Warning: --granularity is not 'sent' or 'clause', the minimum length will be ignored")

        segmenter = get_segmenter(args.granularity, **seg_kwargs)
        format = FORMAT_MAP[args.prompt_format]
        formmatter = LMFormatter(format)
        if args.sum:
            sum_len = args.sum_len
        else:
            sum_len = -1
        controller = LMController(
            segmenter,
            formmatter,
            inference_model,
            output_model,
            hypothesize=args.hypo,
            summarize_len=sum_len,
            retreive_all=args.retrieve_all,
            answer_format=dataset.answer_format,
            logger=logger
        )

    if isinstance(args.output_file, bool):
        if args.output_file: # output file set but not given
            if args.lm:
                g_str = f"chunk-{args.chunk_size}" if args.granularity == "chunk" else args.granularity
                sum_str = f"sum-{args.sum_len}" if args.sum else "no-sum"
                hypo_str = "hypo" if args.hypo else "no-hypo"
                ra_str = "ra" if args.retrieve_all else "no-ra"
                num_q_str = f"{args.num_questions}q"
                output_file: str|None = f"./output/{args.dataset}/lm_{args.infer_model}_{args.out_model}_{args.prompt_format}_{g_str}_{sum_str}_{hypo_str}_{ra_str}_{num_q_str}.json"
                print(f"File name not specified, output to {output_file}")
            else:
                output_file = f"./output/{args.dataset}/baseline_{args.out_model}_{args.num_questions}q.json"
        else:
            output_file = None

    assert output_file is None or output_file.endswith(".json")
    num_questions: int = args.num_questions
    dataset.select(num_questions, randomize=True, seed=42, split='test')
    main(
        controller=controller,
        inference_model=inference_model,
        output_model=output_model,
        dataset=dataset,
        output_file=output_file
    )
