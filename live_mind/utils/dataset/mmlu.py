__all__ = ['MMLUDataset']

import datasets
import re
import random
import logging
from .abc import BaseDataset

MMLU_FORMAT_INST = "Your answer should end with 'The answer is (Choice)'."
MMLU_CATEGORIES: list[str] = [
    'human_sexuality', 'moral_scenarios', 'prehistory', 'astronomy', 'professional_medicine',
    'high_school_macroeconomics', 'international_law', 'high_school_world_history', 'marketing',
    'high_school_geography', 'moral_disputes', 'college_mathematics', 'elementary_mathematics', 
    'high_school_microeconomics', 'anatomy', 'college_medicine', 'business_ethics', 'world_religions', 
    'high_school_statistics', 'medical_genetics', 'human_aging', 'high_school_computer_science', 
    'college_computer_science', 'nutrition', 'sociology', 'us_foreign_policy', 'global_facts', 'philosophy', 
    'high_school_chemistry', 'college_chemistry', 'high_school_biology', 'high_school_psychology', 
    'formal_logic', 'high_school_european_history', 'college_biology', 'miscellaneous', 'professional_law', 
    'high_school_physics', 'jurisprudence', 'clinical_knowledge', 'professional_accounting', 'security_studies', 
    'professional_psychology', 'management', 'abstract_algebra', 'virology', 
    'high_school_government_and_politics', 'high_school_mathematics', 'logical_fallacies', 'public_relations', 
    'college_physics', 'electrical_engineering', 'econometrics', 'computer_security', 'machine_learning', 
    'high_school_us_history', 'conceptual_physics'
]

class MMLUDataset(BaseDataset):
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
        base_num = num // len(MMLU_CATEGORIES)
        num_per_category = {c: base_num for c in MMLU_CATEGORIES}
        for i in range(num % len(MMLU_CATEGORIES)):
            num_per_category[MMLU_CATEGORIES[i]] += 1
        dataset = self.dataset
        assert split in dataset.keys(), f"split {split} is not in the dataset"
        dataset = dataset[split]

        questions = []
        if randomize:
            dataset = dataset.shuffle(seed=seed)
        for c in num_per_category.keys():
            if c not in MMLU_CATEGORIES:
                raise ValueError(f"category {c} is not in the dataset")
            data_by_category = dataset.filter(lambda x: x['subject'] == c)
            if num_per_category[c] == -1:
                questions.extend(data_by_category.to_list())
            elif len(data_by_category) < num_per_category[c]:
                print(f"Warning: only {len(data_by_category)} questions in category {c}, less than the specified number of questions {num_per_category[c]}")
                questions.extend(data_by_category.to_list())
            else:
                questions.extend(data_by_category.to_list()[:num_per_category[c]])
        
        self._selected_questions = questions

    def add_str(self, entry: dict) -> str:
        return " "+self.form_options(entry['choices'])

    def verify_answer(self, response: str, answer_text: int) -> bool:
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
        opts = ['A', 'B', 'C', 'D']
        for opt, o in zip(options, opts):
            option_str += f'({o}): {opt}' + '\n'
        return option_str

    @staticmethod
    def get_prediction(output, guess:bool=False, logger:logging.Logger|None=None) -> int|None:
        """ Extract the prediction from the model output """
        letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        pattern = r"(?:answer) is \(?([ABCDEFGHIJ])\)?"
        match = re.search(pattern, output)
        if match:
            choice = match.group(1)
            if choice not in ['A', 'B', 'C', 'D']:
                if guess:
                    if logger:
                        logger.info(f"in output {output}, {choice} is not a valid choice, do a random guess")
                    return random.choice([1, 2, 3, 4])
                else:
                    if logger:
                        logger.warning(f"in output {output}, {choice} is not a valid choice")
                    return None
            return letter_to_num[match.group(1)]
        else:
            if guess:
                if logger:
                    logger.info(f"in output {output}, extraction failed, do a random guess")
                return random.choice([1, 2, 3, 4])
            else:
                if logger:
                    logger.warning(f"in output {output}, extraction failed")
                return None

