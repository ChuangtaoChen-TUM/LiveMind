""" This script is used to run the solver for real-time latency measure on the MMLU Pro dataset. """
import json
import logging
import pathlib
import argparse
import time
from tqdm import tqdm
from typing import Callable
from live_mind import LMController, CompleteCoTController, BaseController
from live_mind.format.formatter import LMFormatter, CoTFormatter
from live_mind.format import LMFormat
from live_mind.text import (
    TextStreamer,
    nltk_sent_segmenter,
    nltk_comma_segamenter,
    chunk_segmenter,
    char_segmenter
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
    
    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(entry_list, file, indent=4)

SEGMENTER_MAP: dict[str, Callable[[str], list[str]]] = {
    "sentence": nltk_sent_segmenter,
    "clause": nltk_comma_segamenter,
    "chunk": chunk_segmenter,
    "char": char_segmenter
}

FORMAT_MAP = {
    "u_pi": LMFormat.U_PI,
    "u_pli": LMFormat.U_PLI,
    "u_pil": LMFormat.U_PIL,
    "u_ip": LMFormat.U_IP,
    "u_ipl": LMFormat.U_IPL,
    "ua_pil": LMFormat.UA_PIL,
    "u_spi": LMFormat.U_SPI,
    "ua_spi": LMFormat.UA_SPI
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", metavar="str", type=str, default=None, help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("--out_model", metavar="str", type=str, default=None, help="output model, use the inference model if not given")
    parser.add_argument("--prompt_format", metavar="str", type=str, default=None, help=f"prompt format, can be {', '.join(FORMAT_MAP.keys())}")
    parser.add_argument("--granularity", metavar="str", type=str, default=None, help=f"granularity of the text streamer, can be {', '.join(SEGMENTER_MAP.keys())}")
    parser.add_argument("--use_hypo", action="store_true", default=False, help="use hypothesis generation in the LiveMind framework")
    parser.add_argument("--use_sum", action="store_true", default=False, help="use summarization in the LiveMind framework")
    parser.add_argument("--no_retrieve_all", action="store_true", default=False, help="only retrieve one next prompt instead of all available prompts at each step")
    parser.add_argument("--no_lm", action="store_true", default=False, help="disable LiveMind framework, use baseline solver instead")
    parser.add_argument("--output_file", metavar="str", type=str, default=None, help="output results to json file")
    parser.add_argument("--num_questions", metavar="int", type=int, default=100, help="number of questions per category, default: 100")
    parser.add_argument("--log", action="store_true", default=False, help="log the results")
    parser.add_argument("--chunk_size", metavar="int", type=int, default=20, help="chunk size if using chunk granularity")
    parser.add_argument("--sum_len", metavar="int", type=int, default=10, help="summarization length")
    parser.add_argument("--dataset", metavar="str", type=str, default="mmlu_pro", help="dataset to run the solver on, can be mmlu_pro or gsm8k")
    args = parser.parse_args()

    assert args.num_questions == -1 or args.num_questions > 0
    assert args.output_file is None or args.output_file.endswith(".json")

    logger: logging.Logger|None = None
    if args.log:
        # Configure logging
        logger = logging.getLogger("mmlu_pro")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("./output/mmlu_pro/log.log")
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

    if args.no_lm:
        if args.out_model is None:
            raise ValueError("Please specify the output model")
        if args.infer_model is None:
            print("Warning: --no_lm is set, the inference model will be ignored")
        if args.prompt_format is not None:
            print("Warning: --no_lm is set, the prompt format will be ignored")
        if args.granularity is not None:
            print("Warning: --no_lm is set, the granularity will be ignored")
        if args.chunk_size != 20:
            print("Warning: --no_lm is set, the chunk size will be ignored")
        if args.use_hypo:
            print("Warning: --no_lm is set, the hypothesis generation will be ignored")
        if args.use_sum:
            print("Warning: --no_lm is set, the summarization will be ignored")
        if args.no_retrieve_all:
            print("Warning: --no_lm is set, the not retrieve all option will be ignored")
        output_model = get_model(args.out_model)
        if output_model is None:
            raise ValueError("Invalid output model")
        inference_model = output_model
        controller: BaseController = CompleteCoTController(CoTFormatter(), output_model=output_model)
    else:
        if args.infer_model is None:
            raise ValueError("Please specify the inference model")
        if args.out_model is None:
            print("Warning: --out_model is not set, use the inference model as the output model")
        if args.chunk_size != 20 and args.granularity != "chunk":
            print("Warning: --chunk_size is set, but the granularity is not 'chunk', the chunk size will be ignored")
        if not args.use_sum and args.sum_len != 10:
            print("Warning: --use_sum is not set, the summarization length will be ignored")
        inference_model = get_model(args.infer_model)
        if not inference_model:
            raise ValueError("Invalid inference model")
        if args.out_model:
            output_model = get_model(args.out_model)
        else:
            output_model = inference_model
        if args.prompt_format is None:
            raise ValueError("Please specify the prompt format")
        if args.granularity is None:
            raise ValueError("Please specify the granularity")
        segmenter = SEGMENTER_MAP[args.granularity]
        format = FORMAT_MAP[args.prompt_format]
        formmatter = LMFormatter(format)
        if args.use_sum:
            sum_len = args.sum_len
        else:
            sum_len = -1
        controller = LMController(
            segmenter,
            formmatter,
            inference_model,
            output_model,
            hypothesize=args.use_hypo,
            summarize_len=sum_len,
            retreive_all=not args.no_retrieve_all,
            answer_format=dataset.answer_format,
            logger=logger
        )

    num_questions: int = args.num_questions
    output_file: str|None = args.output_file
    dataset.select(num_questions, randomize=True, seed=42, split='test') # choose 100 questions from each category
    main(
        controller=controller,
        inference_model=inference_model,
        output_model=output_model,
        dataset=dataset,
        output_file=output_file
    )
