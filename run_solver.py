""" This script is used to run the solver for real-time latency measure on the MMLU Pro dataset. """
import json
import logging
import pathlib
import argparse
import time
from tqdm import tqdm
from live_mind import LMController, CompleteCoTController, BaseController
from live_mind.action.actions import Infer, Background, Wait, Hypothesize
from live_mind.action.formatter import LMFormatter, CoTFormatter
from live_mind.utils.dataset import MMLUProDataset
from live_mind.utils.text import TextStreamer, nltk_sent_segmenter

from config import BaseModel, MMLU_PRO_PATH, get_model

# Define the action set
CAS = [Infer, Background, Wait, Hypothesize]
SAS = [Infer, Wait]

def main(
    controller: BaseController,
    inference_model: BaseModel,
    output_model: BaseModel,
    dataset: MMLUProDataset,
    output_file: str|None=None,
):
    """ Run the solver on the MMLU Pro dataset and log the results if needed. """
    # warm up the models
    inference_model.chat_complete([{"role": "user", "content": "Hello"}])
    output_model.chat_complete([{"role": "user", "content": "Hello"}])
    entry_list = []

    def delay_fn(text: str): # delay function to simulate the typing speed
        return 0.25*len(text)

    tqdm_bar = tqdm(dataset.selected_questions)
    for entry in tqdm_bar:
        question = entry["question"]
        streamer = TextStreamer(
            question,
            granularity="char", # the granularity can also be set as `chunk`
            delay_fn=delay_fn
        )
        # Initialize the information to be recorded
        total_gen_time: float = 0  # the total time for the model to generate the responses
        usage:dict[str, list] = {
            "completion_tokens": [],
            "prompt_tokens": [],
            "total_tokens": []
        }
        actions = []

        # current input text
        input_text = ""
        gen_time = 0.0
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
            msg = controller(input_text, stream_end=stream_end)

            if msg is None:
                # no update is needed
                continue

            # LLM response
            start_time = time.time()
            if stream_end:
                # use the output model to generate the final response
                response = output_model.chat_complete(msg)
            else:
                response = inference_model.chat_complete(msg)
            gen_time = time.time() - start_time
            total_gen_time += gen_time

            response_text = response["choices"][0]["message"]["content"]
            actions.append(response_text)
            controller.update(response_text)
            usage["completion_tokens"].append(response["usage"]["completion_tokens"])
            usage["prompt_tokens"].append(response["usage"]["prompt_tokens"])
            usage["total_tokens"].append(response["usage"]["total_tokens"])

            if stream_end: # the stream ends
                # overhead_time is the inference model is still generating response when
                # the final text is ready (causing additional latency)
                overhead_time = streamer.current_time - streamer.last_gen_time
                streamer.wait(gen_time) # update current time to the streamer
                controller.reset()
                break

        # latency is defined as the time between the last text is generated and the current time
        latency = streamer.current_time - streamer.last_gen_time
        time_info = {
            "latency": latency,
            "total_gen_time": total_gen_time,
            "overhead_time": overhead_time
        }

        # save info to the entry
        entry["actions"] = actions
        entry["usage"] = usage
        entry["time_info"] = time_info
        entry_list.append(entry)

    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(entry_list, file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", metavar="str", type=str, default="llama-3-8b", help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("--out_model", metavar="str", type=str, default=None, help="output model, use the inference model if not given")
    parser.add_argument("--no_lm", action="store_true", default=False, help="disable LiveMind framework, use baseline solver instead")
    parser.add_argument("--action_set", metavar="str",type=str, default=None, help="action set: CAS, SAS", required=True)
    parser.add_argument("--output_file", metavar="str", type=str, default=None, help="output results to json file")
    parser.add_argument("--num_questions", metavar="int", type=int, default=100, help="number of questions per category, default: 100")
    parser.add_argument("--log", action="store_true", default=False, help="log the results")
    args = parser.parse_args()
    assert args.infer_model in ["llama-3-8b", "llama-3-70b"]
    assert args.num_questions > 0
    assert args.out_model is None or args.out_model in ["llama-3-8b", "llama-3-70b"]
    assert args.action_set in ["CAS", "SAS"]
    assert args.output_file is None or args.output_file.endswith(".json")
    output_file = args.output_file

    logger: logging.Logger|None = None
    if args.log:
        # Configure logging
        logger = logging.getLogger("mmlu_pro")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("./output/mmlu_pro/log.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.action_set == "CAS":
        action_set = CAS
    else:
        action_set = SAS
    if args.no_lm:
        if args.action_set:
            raise ValueError("Cannot specify action set when using baseline solver.")
        controller: BaseController = CompleteCoTController(CoTFormatter())
    else:
        controller = LMController(
            nltk_sent_segmenter,
            LMFormatter(),
            action_set,
            logger=logger
        )

    infer_model_name = args.infer_model
    inference_model = get_model(infer_model_name)
    if args.out_model:
        output_model = get_model(args.out_model)
    else:
        output_model = inference_model
    num_questions: int = args.num_questions
    if not MMLU_PRO_PATH:
        raise ValueError("Please set the path to the MMLU-Pro dataset in config.py")
    mmlu_pro_dataset = MMLUProDataset(MMLU_PRO_PATH)
    mmlu_pro_dataset.select_questions(num_questions, randomize=True) # choose 100 questions from each category
    main(
        controller=controller,
        inference_model=inference_model,
        output_model=output_model,
        dataset=mmlu_pro_dataset,
        output_file=output_file
    )
