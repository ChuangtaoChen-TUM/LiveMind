import datasets
import re
import random
from ...default import MMLU_PRO_PATH

__all__ = ['choose_questions', 'form_options', 'get_prediction']

ALL_CATEGORIES = [
    'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
    'health', 'physics', 'business', 'philosophy', 'economics', 'other',
    'psychology', 'history'
]

def choose_questions(num_per_category: dict|int, randomize=False, seed=42):
    """ Choose questions from the MMLU-Pro dataset
    Args:
        `num_per_category`: `dict` or `int`, the number of questions to choose from each category
            if `int`, the same number of questions will be chosen from each category
        `randomize`: `bool`, whether to shuffle the dataset
        `seed`: `int`, random seed
     """
    if MMLU_PRO_PATH == "":
        raise ValueError("MMLU_PRO_PATH is not set in the default configuration")
    if isinstance(num_per_category, int):
        num_per_category_dict = {}
        for c in ALL_CATEGORIES:
            num_per_category_dict[c] = num_per_category
        num_per_category = num_per_category_dict
    dataset = datasets.load_dataset(MMLU_PRO_PATH)
    dataset = dataset['test']

    questions = []
    if randomize:
        dataset = dataset.shuffle(seed=seed)
    for c in num_per_category.keys():
        if c not in ALL_CATEGORIES:
            raise ValueError(f"category {c} is not in the dataset")
        data_by_category = dataset.filter(lambda x: x['category'] == c)
        if num_per_category[c] == -1:
            questions.extend(data_by_category.to_list())
        elif len(data_by_category) < num_per_category[c]:
            print(f"Warning: only {len(data_by_category)} questions in category {c}, less than the specified number of questions {num_per_category[c]}")
            questions.extend(data_by_category.to_list())
        else:
            questions.extend(data_by_category.to_list()[:num_per_category[c]])

    return questions


def form_options(options: list):
    """ Form the options string for the prompt """
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str

def get_prediction(output, verbose=False, guess=False) -> str:
    """ Extract the prediction from the model output """
    pattern = r"(?:answer) is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        choice = match.group(1)
        if choice not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            if guess:
                if verbose:
                    print(f"{choice} is not a valid choice, do a random guess")
                return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
            else:
                if verbose:
                    print(f"{choice} is not a valid choice")
                return "Not in options"
        return match.group(1)
    else:
        if guess:
            if verbose:
                print(f"extraction failed from {output}, do a random guess")
            return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        else:
            if verbose:
                print(f"extraction failed from {output}")
            return "No match"
