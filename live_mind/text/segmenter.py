""" segmenters are functions that take a string and return a list of strings """
__all__ = [
    "char_segmenter",
    "chunk_segmenter",
    "nltk_sent_segmenter",
    "nltk_comma_segmenter",
    "nltk_word_segmenter",
]

from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Callable


def get_segmenter(name: str, **kwargs) -> Callable[[str], list[str]]:
    match name:
        case "char":
            return char_segmenter

        case "chunk":
            chunk_size = kwargs.get("chunk_size", 10)
            def segmenter(text: str) -> list[str]:
                return chunk_segmenter(text, chunk_size)
            return segmenter

        case "word":
            def segmenter(text: str) -> list[str]:
                return nltk_word_segmenter(text)
            return segmenter

        case "sent":
            min_len = kwargs.get("min_len", 10)
            def segmenter(text: str) -> list[str]:
                return nltk_sent_segmenter(text, min_len)
            return segmenter

        case "clause":
            min_len = kwargs.get("min_len", 10)
            def segmenter(text: str) -> list[str]:
                return nltk_comma_segmenter(text, min_len)
            return segmenter
        case _:
            raise ValueError(f"Unknown segmenter name: {name}")

    
def char_segmenter(text: str) -> list[str]:
    return list(text)


def chunk_segmenter(text: str, chunk_size: int=10) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



def nltk_word_segmenter(text):
    words = word_tokenize(text, preserve_line=True)

    if not words:
        return []

    # Add a space to the start of each word starting with number or letter
    # except the first one
    new_words = [words[0]]

    if len(words) == 1:
        return new_words

    for word in words[1:]:
        if len(word) == 0:
            continue
        if word[0].isalnum():
            new_words.append(" " + word)
        else:
            new_words.append(word)
    
    return new_words


def nltk_sent_segmenter(text: str, min_len: int=10) -> list[str]:
    sents = sent_tokenize(text)
    
    if not sents:
        return []

    # Add a space to the start of each sentence, except the first one
    # merge short sentences into one
    merged_sents = []

    temp_sent = ""
    num_merged_sent = 0
    for sent in sents[:-1]:
        if num_merged_sent == 0:
            temp_sent += sent
        else:
            temp_sent += " " + sent
        num_merged_sent += 1
        if len(temp_sent) >= min_len:
            merged_sents.append(temp_sent)
            temp_sent = " "
            num_merged_sent = 0

    if num_merged_sent == 0:
        last_sent = temp_sent + sents[-1]
    else:
        last_sent = temp_sent + " " + sents[-1]
    merged_sents.append(last_sent)
    return merged_sents


def nltk_comma_segmenter(text: str, min_len: int=10) -> list[str]:
    sents: list[str] = nltk_sent_segmenter(text, min_len=min_len)
    # Add a comma and a space to the end of each sentence, except the last one
    segs = []
    for sent in sents:
        segs += _split_by_commas(sent, min_len)
    return segs


def _split_by_commas(text: str, min_len:int=10) -> list[str]:
    clauses = text.split(",")
    last_is_comma = False
    if len(clauses) > 0 and clauses[-1] == "":
        last_is_comma = True
        clauses = clauses[:-1]

    merged_clauses = []
    temp_clause = ""

    for index in range(len(clauses)-1):
        temp_clause += clauses[index] + ","
        if len(temp_clause) >= min_len and not _check_num_chars(clauses[index], clauses[index+1]):
            # if reached the min length and no consecutive digits
            merged_clauses.append(temp_clause)
            temp_clause = ""

    last_clause = temp_clause + clauses[-1]
    if last_is_comma:
        last_clause += ","

    merged_clauses.append(last_clause)
    return merged_clauses

def _check_num_chars(text_1: str, text_2: str) -> bool:
    """ Check if the last character of `text_1` and the first character of `text_2` are both digits """
    text_1 = text_1.strip()
    text_2 = text_2.strip()

    if not text_1 or not text_2:
        return False
    
    return text_1[-1].isdigit() and text_2[0].isdigit()
