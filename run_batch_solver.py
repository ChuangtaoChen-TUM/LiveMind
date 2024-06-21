""" this script is used to run the batched solver on the MMLU-Pro dataset """
import json
import logging
import pathlib
import argparse
from alive_progress import alive_bar
from live_mind.solver.batch_solver import batch_solver, batch_solver_base
from live_mind.utils.dataset import batch_generator, sent_len
from live_mind.utils.dataset.mmlu_pro import (
    choose_questions,
    form_options,
    get_prediction,
)
from live_mind.default import MMLU_FORMAT, get_model
from live_mind import default

logging.basicConfig(
    filename='./output/mmlu_pro/log.log',  # Log file name
    level=logging.INFO,   # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def main(
    dataset: list = [],
    batch_size: int = 32,
    output_file = None,
    solver=None,
    model=None,
    assist_model=None
):
    per_category_accuracy = {}
    success, fail = 0, 0

    entry_list = []

    with alive_bar(-(-len(dataset)//batch_size)) as bar:
        for entries in batch_generator(dataset, batch_size):
            questions = [
                entry["question"] for entry in entries
            ]
            final_lines = [
                form_options(entry["options"]) for entry in entries
            ]
            texts, infos = solver(
                questions,
                model=model,
                answer_format=MMLU_FORMAT,
                assist_model=assist_model,
                final_lines=final_lines
            )
            for i in range(len(entries)):
                entry = entries[i]
                entry["solution"] = texts[i]
                entry.update(infos[i])
                prediction = get_prediction(texts[i], verbose=False, guess=False)
                if per_category_accuracy.get(entry["category"]) is None:
                    per_category_accuracy[entry["category"]] = {"success": 0, "fail": 0}
                if entry["answer"] == prediction:
                    success += 1
                    per_category_accuracy[entry["category"]]["success"] += 1
                else:
                    fail += 1
                    per_category_accuracy[entry["category"]]["fail"] += 1
                entry_list.append(entry)
            logging.info(f"{success / (success + fail):.3f}")
            bar.text(f"{success / (success + fail):.3f}")
            bar()

    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(entry_list, file, indent=4)

    for k, v in per_category_accuracy.items():
        if v["success"] + v["fail"] == 0:
            print(f"Category {k}: no data")
        else:
            print(f"Category {k}: {v['success'] / (v['success'] + v['fail']):.3f}")
    print(f"Total accuracy: {success / (success + fail):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", metavar="str", type=str, default="llama-3-8b", help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("--use_lm", action="store_true", default=False, help="whether to use LiveMind or baseline inference")
    parser.add_argument("--batch_size", metavar="int", type=int, default=128, help="batch size")
    parser.add_argument("--assist_model", metavar="str", type=str, default=None, help="output model, use the inference model if not given")
    parser.add_argument("--action_set", metavar="str",type=str, default="CAS", help="action set: CAS, SAS")
    parser.add_argument("--output_file", metavar="str", type=str, default=None, help="output results to json file")
    args = parser.parse_args()
    assert args.model in ["llama-3-8b", "llama-3-70b"]
    assert args.batch_size > 0
    assert args.assist_model is None or args.assist_model in ["llama-3-8b", "llama-3-70b"]
    assert args.action_set in ["CAS", "SAS"]
    assert args.output_file is None or args.output_file.endswith(".json")

    model_name = args.model
    model = get_model(model_name)
    if args.assist_model:
        assist_model = get_model(args.assist_model)
    else:
        assist_model = None
    dataset = choose_questions(-1)
    dataset.sort(key=lambda x: sent_len(x["question"])) # sort by sentence length to make the batched solver more efficient

    default.USE_COMP = args.action_set == "CAS" # choose action set
    solver = batch_solver if args.use_lm else batch_solver_base
    batch_size = args.batch_size
    output_file = args.output_file
    main(
        dataset=dataset,
        batch_size=batch_size,
        output_file=output_file,
        solver=batch_solver,
        model=model,
        assist_model=assist_model
    )
