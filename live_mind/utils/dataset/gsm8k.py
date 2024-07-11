import datasets
import re
from .abc import BaseDataset

GSM8k_CATEGORIES = []

class GSM8kDataset(BaseDataset):
    """ The GSM-8k dataset """
    def __init__(self, path):
        self.path = path
        self.dataset = datasets.load_dataset(path)
        self._selected_questions = []
        self._answer_format = "The answer is an integer or decimal number. Your answer must end with `The answer is: (number)`"
        self.answer_re = re.compile(r"#### (\-?[0-9\.\,]+)")
        self.response_re = re.compile(r"[Tt]he answer is: \$?(\-?[0-9\.\,]+)") # some answer may have a dollar sign

    def select(
        self,
        num: int,
        randomize:bool=False,
        seed:int=42,
        split:str='test'
    ):
        """ Choose questions from the GSM-8k dataset, this action will update `self.selected_questions`

        Args:
        - `num`: `int`, the number of questions to select
        - `randomize`: `bool`, whether to shuffle the dataset
        - `seed`: `int`, random seed
        """
        dataset = self.dataset
        assert split in dataset.keys(), f"split {split} is not in the dataset"
        dataset = dataset[split]

        questions = []
        if randomize:
            dataset = dataset.shuffle(seed=seed)
        assert num == -1 or num > 0, f"Invalid number of questions: {num}"
        if num == -1:
            questions = dataset.to_list()
        elif len(dataset) < num:
            print(f"Warning: only {len(dataset)} questions in the dataset, less than the specified number of questions {num}")
            questions = dataset.to_list()
        else:
            questions = dataset.to_list()[:num]

        self._selected_questions = questions

    def verify_answer(self, response: str, answer_text: str):
        match_ans = self.answer_re.search(answer_text)
        if match_ans:
            gt_answer = match_ans.group(1).replace(",", "").strip(".")
        else:
            raise ValueError(f"Failed to extract the answer from the text: {answer_text}")
        match_response = self.response_re.search(response)
        if match_response:
            pred_answer = match_response.group(1).replace(",", "").strip(".") # remove the comma
        else:
            pred_answer = None
        if pred_answer is None:
            print(f"Failed to extract the answer from the response: {response}")
            return False
        try:
            pred_float_value = float(pred_answer)
            gt_float_value = float(gt_answer)
            return pred_float_value == gt_float_value
        except ValueError:
            print(f"Failed to convert the answer {pred_answer} to float")
            return False

    def add_str(entry: dict) -> str:
        return ""

    @property
    def selected_questions(self):
        return self._selected_questions

    @property
    def answer_format(self) -> str:
        return self._answer_format
