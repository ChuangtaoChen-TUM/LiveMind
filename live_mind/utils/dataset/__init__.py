import nltk

__all__ = ['mmlu_pro', 'batch_generator', 'sent_len']

from . import mmlu_pro

def batch_generator(iterable, batch_size, drop_last=False):
    """
    A generator function that yields batches of elements from an iterable.

    Args:
        iterable (iterable): The input iterable (e.g., list, generator).
        batch_size (int): The number of elements in each batch.

    Yields:
        list: A list of elements of size `batch_size`, except for the last
              batch which may be smaller if there are not enough elements.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch

def sent_len(text) -> int:
    return len(nltk.sent_tokenize(text))
