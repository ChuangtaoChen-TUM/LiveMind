""" segmenters are functions that take a string and return a list of strings """
__all__ = [
    "char_segmenter",
    "chunk_segmenter",
    "nltk_sent_segmenter",
    "nltk_comma_segamenter"
]

from nltk.tokenize import sent_tokenize
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

        case "sent":
            min_len = kwargs.get("min_len", 10)
            def segmenter(text: str) -> list[str]:
                return nltk_sent_segmenter(text, min_len)
            return segmenter

        case "clause":
            min_len = kwargs.get("min_len", 10)
            def segmenter(text: str) -> list[str]:
                return nltk_comma_segamenter(text, min_len)
            return segmenter
        case _:
            raise ValueError(f"Unknown segmenter name: {name}")

    
def char_segmenter(text: str) -> list[str]:
    return list(text)


def chunk_segmenter(text: str, chunk_size: int=10) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def nltk_sent_segmenter(text: str, min_len: int=10) -> list[str]:
    sents = sent_tokenize(text)
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


def nltk_comma_segamenter(text: str, min_len: int=10) -> list[str]:
    sents: list[str] = nltk_sent_segmenter(text, min_len=min_len)
    # Add a comma and a space to the end of each sentence, except the last one
    segs = []
    for sent in sents:
        segs += _split_by_commas(sent, min_len)
    return segs


def _split_by_commas(text: str, min_len:int=10) -> list[str]:
    clauses = text.split(",")
    merged_clauses = []
    temp_clause = ""
    for clause in clauses[:-1]:
        temp_clause += clause + ","
        if len(temp_clause) >= min_len:
            merged_clauses.append(temp_clause)
            temp_clause = ""
    last_clause = temp_clause + clauses[-1]
    merged_clauses.append(last_clause)
    return merged_clauses
