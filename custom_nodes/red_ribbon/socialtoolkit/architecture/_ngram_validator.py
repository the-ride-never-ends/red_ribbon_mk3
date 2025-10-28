from collections import deque
from itertools import chain, islice
from typing import Any, Iterable, Literal, Optional


import spacy


from ._errors import InitializationError


try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise InitializationError(f"Failed to load spaCy model: {e}") from e




def pad_sequence(
    sequence: Iterable[Any],
    n: int,
    pad_left: bool = False,
    pad_right: bool = False,
    left_pad_symbol: Optional[str] = None,
    right_pad_symbol: Optional[str] = None,
) -> Iterable[Any]:
    """Returns a padded sequence of items before ngram extraction.

    Args:
        sequence: The source data to be padded (sequence or iter).
        n: The degree of the ngrams.
        pad_left: Whether the ngrams should be left-padded.
        pad_right: Whether the ngrams should be right-padded.
        left_pad_symbol: The symbol to use for left padding (default is None).
        right_pad_symbol: The symbol to use for right padding (default is None).

    Returns:
        Padded sequence or iter.

    Examples:
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


# NOTE: These are directly copy-pasted from nltk.util to avoid adding nltk as a dependency.
def ngrams(sequence: Iterable[Any], n, **kwargs):
    """Return the ngrams generated from a sequence of items, as an iterator.

    Args:
        sequence: The source data to be converted into ngrams.
        n: The degree of the ngrams.
        pad_left: Whether the ngrams should be left-padded.
        pad_right: Whether the ngrams should be right-padded.
        left_pad_symbol: The symbol to use for left padding (default is None).
        right_pad_symbol: The symbol to use for right padding (default is None).

    Yields:
        Tuple of n consecutive items from the sequence.

    Example:
        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

        >>> # Wrap with list for a list version of this function. Set pad_left
        >>> # or pad_right to true in order to get additional ngrams:
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    """
    sequence = pad_sequence(sequence, n, **kwargs)

    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    # https://docs.python.org/3/library/itertools.html?highlight=sliding_window#itertools-recipes
    it = iter(sequence)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

class NgramValidator:

    _STOP_WORDS = nlp.Defaults.stop_words
    _PUNCTUATION = {
        '.', ',', '!', '?', ';', ':', '...'
        '-', '_', '(', ')', '[', ']', '{', '}', 
        '"', "'","''", "``"
    }

    def __init__(self, resources, configs):
        self.resources = resources
        self.configs = configs

        self.ngrams = ngrams

    def _check_word(self, word: str) -> bool:
        """True if a word is not a stop word or a punctuation."""
        return word not in self._STOP_WORDS and word not in self._PUNCTUATION

    def _filtered_words(self, sentence: str) -> list[str]:
        """Return list of words in sentence after filtering out stop words and punctuation."""
        doc = nlp(sentence)
        return [token.text for token in doc if self._check_word(token.text)]

    def text_to_ngrams(self, text: str, n: int) -> list[str]:
        """Convert text to list of n-grams."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        doc = nlp(text)
        ngrams = [
            gram for sent in doc.sents 
            for gram in self.ngrams(self._filtered_words(sent.text), n)
        ]
        return ngrams

    def sentence_ngram_fraction(self, original_text: str, test_text: str, n: int) -> float | Literal[True]:
        """Calculate fraction of sentence ngrams from test text found in original text.

        Args:
            original_text (str): Original text. Is a superset of test_text.
            test_text (str): Test text. Is a subset of original_text.
            n (int): Number of words to include per ngram.

        Returns:
            float | Literal[True]: Fraction of ngrams from test_text that were found in
            original_text. Returns True if test_text has no ngrams.

        Raises:
            TypeError: If original_text or test_text is not a string, or if n is
                not an integer.
            ValueError: If n is not a positive integer.
            RuntimeError: If ngram extraction fails for either text.
        """
        for var in [("original_text", original_text), ("test_text", test_text)]:
            if not isinstance(var[1], str):
                raise TypeError(f"{var[0]} must be a string, got {type(var[1]).__name__}")
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

        try:
            test_ngrams = self.text_to_ngrams(test_text, n)
        except Exception as e:
            raise RuntimeError(f"Failed to extract ngrams from test_text: {e}") from e
        num_test_ngrams = len(test_ngrams)
        if not num_test_ngrams:
            return True

        try:
            original_ngrams = set(self.text_to_ngrams(original_text, n))
        except Exception as e:
            raise RuntimeError(f"Failed to extract ngrams from original_text: {e}") from e
        num_ngrams_found = sum(ngram in original_ngrams for ngram in test_ngrams)
        return num_ngrams_found / num_test_ngrams