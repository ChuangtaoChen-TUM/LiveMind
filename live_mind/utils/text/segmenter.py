""" segmenters are functions that take a string and return a list of strings """
__all__ = [
    "nltk_sent_segmenter",
    "split_segmenter"
]

from nltk.tokenize import sent_tokenize


def nltk_sent_segmenter(text: str) -> list[str]:
    sents = sent_tokenize(text)
    # Add a space to the end of each sentence, except the last one
    if len(sents) > 1:
        return [sent+" " for sent in sents[:-1]] + [sents[-1]]
    return sents

def split_segmenter(text: str) -> list[str]:
    words = text.split()
    # Add a space to the end of each word, except the last one
    if len(words) > 1:
        return [word+" " for word in words[:-1]] + [words[-1]]
    return words
