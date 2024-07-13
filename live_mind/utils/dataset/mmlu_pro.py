import datasets
import re
import random
import logging
from .abc import BaseDataset

__all__ = ['MMLUProDataset', 'MMLU_FORMAT_INST']

MMLU_FORMAT_INST = "Your answer should end with 'The answer is (Choice)'."
MMLU_PRO_CATEGORIES = [
    'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
    'health', 'physics', 'business', 'philosophy', 'economics', 'other',
    'psychology', 'history'
]

class MMLUProDataset(BaseDataset):
    """ The MMLU-Pro dataset """
    def __init__(self, path):
        self.path = path
        self.dataset = datasets.load_dataset(path)
        self._selected_questions = []
        self._answer_format = MMLU_FORMAT_INST

    def select(
        self,
        num: int,
        randomize:bool=False,
        seed:int=42,
        split:str='test'
    ):
        """ Choose questions from the MMLU-Pro dataset, this action will update `self.selected_questions`
        
        Args:
        - `num_per_category`: `dict` or `int`, the number of questions to select from each category
            if `int`, the same number of questions will be selected from each category
        - `randomize`: `bool`, whether to shuffle the dataset
        - `seed`: `int`, random seed
        """
        if isinstance(num, int):
            base_num = num // len(MMLU_PRO_CATEGORIES)
            num_per_category = {c: base_num for c in MMLU_PRO_CATEGORIES}
            for i in range(num % len(MMLU_PRO_CATEGORIES)):
                num_per_category[MMLU_PRO_CATEGORIES[i]] += 1
        dataset = self.dataset
        assert split in dataset.keys(), f"split {split} is not in the dataset"
        dataset = dataset[split]

        questions = []
        if randomize:
            dataset = dataset.shuffle(seed=seed)
        for c in num_per_category.keys():
            if c not in MMLU_PRO_CATEGORIES:
                raise ValueError(f"category {c} is not in the dataset")
            data_by_category = dataset.filter(lambda x: x['category'] == c)
            if num_per_category[c] == -1:
                questions.extend(data_by_category.to_list())
            elif len(data_by_category) < num_per_category[c]:
                print(f"Warning: only {len(data_by_category)} questions in category {c}, less than the specified number of questions {num_per_category[c]}")
                questions.extend(data_by_category.to_list())
            else:
                questions.extend(data_by_category.to_list()[:num_per_category[c]])

        self._selected_questions = questions

    def add_str(self, entry: dict) -> str:
        return " "+self.form_options(entry['options'])

    def verify_answer(self, response: str, answer_text: str):
        prediction = self.get_prediction(response)
        return prediction == answer_text

    @property
    def selected_questions(self):
        return self._selected_questions

    @property
    def answer_format(self) -> str:
        return self._answer_format

    @staticmethod
    def form_options(options: list) -> str:
        """ Form the options string for the prompt """
        option_str = 'Options are:\n'
        opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for opt, o in zip(options, opts):
            option_str += f'({o}): {opt}' + '\n'
        return option_str

    @staticmethod
    def get_prediction(output, guess:bool=False, logger:logging.Logger|None=None) -> str|None:
        """ Extract the prediction from the model output """
        pattern = r"(?:answer) is \(?([ABCDEFGHIJ])\)?"
        match = re.search(pattern, output)
        if match:
            choice = match.group(1)
            if choice not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                if guess:
                    if logger:
                        logger.info(f"in output {output}, {choice} is not a valid choice, do a random guess")
                    return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
                else:
                    if logger:
                        logger.warning(f"in output {output}, {choice} is not a valid choice")
                    return None
            return match.group(1)
        else:
            if guess:
                if logger:
                    logger.info(f"in output {output}, extraction failed, do a random guess")
                return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
            else:
                if logger:
                    logger.warning(f"in output {output}, extraction failed")
                return None

