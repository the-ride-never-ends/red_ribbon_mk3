# -*- coding: utf-8 -*-
"""ELM Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to
a given variable (e.g. parking requirements, wind ordinances, etc.).
"""
from collections import defaultdict
import logging
from pathlib import Path
from typing import DefaultDict


from pydantic import BaseModel, ValidationError
import yaml


logger = logging.getLogger(__name__)


def _load_keywords() -> DefaultDict[str, dict]:
    """Load keywords from YAML files in the words directory."""
    class _HeuristicWords(BaseModel):
        NOT_WORDS: list[str]
        GOOD_KEYWORDS: list[str]
        GOOD_ACRONYMS: list[str]
        GOOD_ACRONYM_CONTEXTS: list[str]
        GOOD_PHRASES: list[str]

    paths = {
        file for file in (Path(__file__).parent / "words").glob("*.yaml")
    }
    words_from_yaml = defaultdict(dict)

    for path in paths:
        name = path.stem
        try:
            with open(path, "r") as f:
                words = dict(yaml.safe_load(f)) 
                words_from_yaml[name] = _HeuristicWords(**words).model_dump()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YAML file for variable '{name}' not found: {e}")
        except yaml.YAMLError as e:
            raise IOError(f"Error parsing YAML file for variable '{name}': {e}")
        except ValidationError as e:
            raise ValueError(f"Validation error loading YAML file for variable '{name}': {e}")
        except Exception as e:
            raise IOError(f"Unexpected error loading YAML file for variable '{name}': {e}")
    return words_from_yaml


_WORDS_FROM_YAML_FILES = _load_keywords()


class _PossiblyMentions:

    def __init__(
            self,
            *,
            variable: str,
            # NOTE We explicitly define these here for intellisense to pick up the expected YAML keys
            expected_keys: set[str] = { 
                "NOT_WORDS",
                "GOOD_KEYWORDS",
                "GOOD_ACRONYMS",
                "GOOD_ACRONYM_CONTEXTS",
                "GOOD_PHRASES",
            },
            available_heuristic_words: DefaultDict = _WORDS_FROM_YAML_FILES
        ) -> None:
        words = {}
        if variable not in available_heuristic_words:
            raise ValueError(f"variable '{variable}' lacks a corresponding YAML file.")
        else:
            words = available_heuristic_words[variable]

        if not all(key in words for key in expected_keys):
            raise KeyError(f"YAML file for variable '{variable}' is missing one or more expected keys: {expected_keys}")

        self._NOT_WORDS = words["NOT_WORDS"]
        self._GOOD_KEYWORDS = words["GOOD_KEYWORDS"]
        self._GOOD_ACRONYMS = words["GOOD_ACRONYMS"]
        self._GOOD_ACRONYM_CONTEXTS = words["GOOD_ACRONYM_CONTEXTS"]
        self._GOOD_PHRASES = words["GOOD_PHRASES"]

    def convert_to_heuristics_text(self, text):
        """Convert text for heuristic parking content parsing"""
        heuristics_text = text.casefold()
        for word in self._NOT_WORDS:
            heuristics_text = heuristics_text.replace(word, "")
        return heuristics_text

    def count_single_keyword_matches(self, heuristics_text):
        """Count number of good parking keywords that appear in text."""
        return sum(keyword in heuristics_text for keyword in self._GOOD_KEYWORDS)

    def count_acronym_matches(self, heuristics_text):
        """Count number of good parking acronyms that appear in text."""
        acronym_matches = 0
        for context in self._GOOD_ACRONYM_CONTEXTS:
            acronym_keywords = {
                context.format(acronym=acronym) for acronym in self._GOOD_ACRONYMS
            }
            acronym_matches = sum(
                keyword in heuristics_text for keyword in acronym_keywords
            )
            if acronym_matches > 0:
                break
        return acronym_matches

    def count_phrase_matches(self, heuristics_text):
        """Count number of good parking phrases that appear in text."""
        return sum(
            all(keyword in heuristics_text for keyword in phrase.split(" "))
            for phrase in self._GOOD_PHRASES
        )



def possibly_mentions(text: str, match_count_threshold: int = 1, variable: str = "wind") -> bool:
    """Perform a heuristic check for mention of a variable and terms related to the variable in text.

    This check first strips the text of any parking "look-alike" words
    (e.g. "parking lot", "parkway", etc). Then, it checks for particular
    keywords, acronyms, and phrases that pertain to parking requirements
    in the text. If enough keywords are mentions (as dictated by
    `match_count_threshold`), this check returns ``True``.

    Args:
        text: Input text that may or may not mention parking in relation to
            parking minimums.
        match_count_threshold: Number of keywords that must match for the text to pass this
            heuristic check. Count must be strictly greater than this value.
            Defaults to 1.
        variable: The variable type to check for (e.g., "wind", "parking"). Must have a
            corresponding YAML file in the words directory. Defaults to "wind".

    Returns:
        True if the number of keywords/acronyms/phrases detected
        exceeds the `match_count_threshold`.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(variable).__name__}")
    if not text:
        raise ValueError("text cannot be empty")
    if match_count_threshold < 0:
        raise ValueError("match_count_threshold must be non-negative")
    if not isinstance(variable, str):
        raise TypeError(f"variable must be a string, got {type(variable).__name__}")
    if not variable:
        raise ValueError("variable cannot be empty")

    pm = _PossiblyMentions(variable=variable)
    heuristics_text = pm.convert_to_heuristics_text(text)
    total_keyword_matches = pm.count_single_keyword_matches(heuristics_text)
    total_keyword_matches += pm.count_acronym_matches(heuristics_text)
    total_keyword_matches += pm.count_phrase_matches(heuristics_text)
    return total_keyword_matches > match_count_threshold
