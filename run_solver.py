""" This script is used to run the solver for real-time latency measure on the MMLU Pro dataset. """
import json
import logging
import pathlib
import argparse
from alive_progress import alive_bar
from live_mind.solver.solver import solver as lm_solver, solver_base
from live_mind.utils.dataset.mmlu_pro import (
    choose_questions,
    form_options,
    get_prediction
)
from live_mind import default
from live_mind.default import MMLU_FORMAT, get_model

# Configure logging
logging.basicConfig(
    filename='error.log',  # Log file name
    level=logging.INFO,   # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def main(
    dataset: list = [],
    output_file=None,
    solver=None,
    model=None,
    assist_model=None
):
    per_category_accuracy = {}
    success, fail = 0, 0

    entry_list = []

    with alive_bar(len(dataset)) as bar:
        for entry in dataset:
            question = entry["question"]
            final_line = form_options(entry["options"])
            response, usage, time_info = solver(
                question,
                model=model,
                answer_format=MMLU_FORMAT,
                assist_model=assist_model,
                final_line=final_line
            )
            entry["solution"] = response
            entry["usage"] = usage
            entry["time_info"] = time_info
            prediction = get_prediction(response, verbose=False, guess=False)
            if per_category_accuracy.get(entry["category"]) is None:
                per_category_accuracy[entry["category"]] = {"success": 0, "fail": 0}
            if entry["answer"] == prediction:
                success += 1
                per_category_accuracy[entry["category"]]["success"] += 1
            else:
                fail += 1
                per_category_accuracy[entry["category"]]["fail"] += 1
            entry_list.append(entry)
            # logging.info(f"{success / (success + fail):.3f}")
            bar.text(f"{success / (success + fail):.3f}")
            bar()

    if output_file:
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(entry_list, file, indent=4)

    for k, v in per_category_accuracy.items():
        if v["success"] + v["fail"] == 0:
            print(f"{k}: no data")
        else:
            print(f"{k}: {v['success'] / (v['success'] + v['fail']):.3f}")

    print(f"Total accuracy: {success / (success + fail):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", metavar="str", type=str, default="llama-3-8b", help="inference model: llama-3-8b, llama-3-70b")
    parser.add_argument("--use_lm", action="store_true", default=False, help="whether to use LiveMind or baseline inference")
    parser.add_argument("--assist_model", metavar="str", type=str, default=None, help="output model, use the inference model if not given")
    parser.add_argument("--action_set", metavar="str",type=str, default="CAS", help="action set: CAS, SAS")
    parser.add_argument("--output_file", metavar="str", type=str, default=None, help="output results to json file")
    parser.add_argument("--num_questions", metavar="int", type=int, default=100, help="number of questions per category, default: 100")
    args = parser.parse_args()
    assert args.model in ["llama-3-8b", "llama-3-70b"]
    assert args.num_questions > 0
    assert args.assist_model is None or args.assist_model in ["llama-3-8b", "llama-3-70b"]
    assert args.action_set in ["CAS", "SAS"]
    assert args.output_file is None or args.output_file.endswith(".json")

    model_name = args.model
    model = get_model(model_name)
    if args.assist_model:
        assist_model = get_model(args.assist_model)
    else:
        assist_model = None
    num_questions = args.num_questions
    dataset = choose_questions(num_questions, randomize=True) # choose 100 questions from each category
    default.USE_COMP = args.action_set == "CAS" # choose action set
    solver = lm_solver if args.use_lm else solver_base
    output_file = args.output_file
    main(
        dataset=dataset,
        output_file=output_file,
        solver=solver,
        model=model,
        assist_model=assist_model
    )
