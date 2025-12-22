# -*- coding: utf-8 -*-
"""ELM Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""
import asyncio
import logging
from typing import Any, DefaultDict


from elm import ApiBase
from elm.ords.utilities.parsing import merge_overlapping_texts


from .content import ValidationWithMemory
from .heuristics import possibly_mentions


class OrdinanceValidator:
    """Check document text for wind ordinances

    .. start desc ov
    Purpose:
        Determine wether a document contains relevant ordinance
        information.
    Responsibilities:
        1. Determine wether a document contains relevant (e.g.
        utility-scale wind zoning) ordinance information by splitting
        the text into chunks and parsing them individually using LLMs.
    Key Relationships:
        Child class of
        :class:`~elm.ords.validation.content.ValidationWithMemory`,
        which allows the validation to look at neighboring chunks of
        text.

    .. end desc
    """
    def __init__(self, *, resources: dict[str, Any]) -> None:
        """Initialize the OrdinanceValidator.

        Args:
            resources (dict[str, Any]): Dictionary containing required resources including:
                - logger: Logger instance for logging operations.
                - validation_with_memory: ValidationWithMemory instance that handles text chunk
                  validation. This validator may refer to previous text chunks to answer
                  validation questions.
                - prompt_dict: Dictionary containing prompt configurations.
        """
        self.resources = resources

        self.logger: logging.Logger = self.resources['logger']
        self.validation_with_memory: ValidationWithMemory = self.resources['validation_with_memory']
        self.prompt_dict: DefaultDict = self.resources['prompt_dict']['ORDINANCE_VALIDATOR']

        # Get is_legal_text prompt and key
        _is_legal_text_file_contents = self.prompt_dict.pop("is_legal_text")
        self._is_legal_text_prompt: str = _is_legal_text_file_contents["prompt"]
        self._is_legal_text_key: str = _is_legal_text_file_contents["key"]

        self._legal_text_mem = []
        self._wind_mention_mem = []
        self._ordinance_chunks = []

    @property
    def is_legal_text(self) -> bool:
        """True if text was found to be from a legal source, else False."""
        if not self._legal_text_mem:
            return False
        return sum(self._legal_text_mem) >= 0.5 * len(self._legal_text_mem)

    @property
    def ordinance_text(self) -> str:
        """Combined ordinance text from the individual chunks."""
        idxs_to_grab = set()
        for info in self._ordinance_chunks:
            idxs_to_grab |= { # NOTE |= : union (add elements)
                info["idx"] + x for x in range(1 - self.validation_with_memory.num_to_recall, 2)
            }

        text = [
            self.validation_with_memory.text_chunks[idx]
            for idx in sorted(idxs_to_grab)
            if 0 <= idx < len(self.validation_with_memory.text_chunks)
        ]
        return merge_overlapping_texts(text)

    async def parse(self, min_chunks_to_process: int = 3, variable: str ="wind"):
        """Parse text chunks and look for ordinance text.

        Args:
            min_chunks_to_process (int, optional): Minimum number of chunks to process before checking if
                document resembles legal text and ignoring chunks that don't
                pass the wind heuristic. Defaults to 3.
            variable (str, optional): Variable to check for mentions. Defaults to "wind".

        Returns:
            bool: True if any ordinance text was found in the chunks, else False.

        Raises:
            TypeError: If min_chunks_to_process is not an int or variable is not a str.
            ValueError: If min_chunks_to_process is not a positive integer or variable is empty or whitespace.
            KeyError: If variable is not found in prompt_dict dictionary attribute.
        """
        if not isinstance(min_chunks_to_process, int):
            raise TypeError(f"min_chunks_to_process must be an int, got {type(min_chunks_to_process).__name__}")
        if not isinstance(variable, str):
            raise TypeError(f"variable must be a str, got {type(variable).__name__}")
        if min_chunks_to_process <= 0:
            raise ValueError(f"min_chunks_to_process must be a positive integer, got {min_chunks_to_process}")

        variable = variable.strip()
        if variable not in self.prompt_dict:
            raise KeyError(f"variable '{variable}' not found in prompt_dict")

        for idx, text in enumerate(self.validation_with_memory.text_chunks):
            self._wind_mention_mem.append(possibly_mentions(text, variable=variable))
            if idx >= min_chunks_to_process:
                if not self.is_legal_text:
                    return False

                # fmt: off
                if not any(self._wind_mention_mem[-self.validation_with_memory.num_to_recall:]):
                    continue

            self.logger.debug(f"Processing text at idx {idx}")
            self.logger.debug(f"Text:\n{text}")

            if idx < min_chunks_to_process:
                is_legal_text = await self.validation_with_memory.parse_from_ind(
                    idx, self._is_legal_text_prompt, key=self._is_legal_text_key
                )
                self._legal_text_mem.append(is_legal_text)
                if not is_legal_text:
                    self.logger.debug(f"Text at idx {idx} is not legal text")
                    continue
                self.logger.debug(f"Text at idx {idx} is legal text")

            for name, prompt_dict in self.prompt_dict.items():
                result = await self.validation_with_memory.parse_from_ind(
                    idx, prompt_dict["prompt"], key=prompt_dict["key"]
                )
                if result is False:
                    self.logger.debug(f"Text at idx {idx} did not pass prompt '{name}'")
                else:
                    self.logger.debug(f"Text at idx {idx} passed prompt '{name}'")

            self._ordinance_chunks.append({"text": text, "idx": idx})
            self.logger.debug(f"Added text at idx {idx} to ordinances")
            # mask, since we got a good result
            self._wind_mention_mem[-1] = False

        return bool(self._ordinance_chunks)


class OrdinanceExtractor:
    """Extract succinct ordinance text from input

    .. start desc oe
    Purpose:
        Extract relevant ordinance text from document.
    Responsibilities:
        1. Extract portions from chunked document text relevant to
           particular ordinance type (e.g. wind zoning for utility-scale
           systems).
    Key Relationships:
        Uses a :class:`~elm.ords.llm.calling.StructuredLLMCaller` for
        LLM queries.

    .. end desc
    """

    def __init__(self, resources: dict[str, Any]) -> None:
        """Initialize the OrdinanceExtractor.

        Parameters:
            resources (dict[str, Any]): Dictionary containing the following keys:
                - llm_caller: LLM Caller instance used to extract ordinance info with.
                - logger: Logger instance for logging operations.
                - prompt_dict: Dictionary containing prompt configurations.
                - semaphore: Semaphore for concurrency control.
        """
        self.resources = resources

        self.logger: logging.Logger = self.resources['logger']
        self.llm_caller = self.resources['llm_caller']
        self.prompt_dict = self.resources['prompt_dict']['ORDINANCE_EXTRACTOR']
        self._semaphore = self.resources['semaphore']

        self._MODEL_INSTRUCTIONS_RESTRICTIONS = self.prompt_dict["MODEL_INSTRUCTIONS_RESTRICTIONS"]
        self._MODEL_INSTRUCTIONS_SIZE = self.prompt_dict["MODEL_INSTRUCTIONS_SIZE"]
        self._SYSTEM_MESSAGE = self.prompt_dict["SYSTEM_MESSAGE"]


    async def _run_task_with_limit(self, task):
        async with self._semaphore:
            return await task


    async def _process(self, text_chunks, instructions, valid_chunk):
        """Perform extraction processing."""
        self.logger.info(f"Extracting ordinance text from {len(text_chunks)} text chunks asynchronously...")

        current_task = asyncio.current_task()
        if current_task is None:
            raise asyncio.InvalidStateError("No current asyncio task found")
        outer_task_name = current_task.get_name()

        tasks = [
            asyncio.create_task(
                self.llm_caller.call(
                    sys_msg=self._SYSTEM_MESSAGE,
                    content=f"Text:\n{chunk}\n{instructions}",
                    usage_sub_label="document_ordinance_summary",
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        # Add a concurrency limiter
        summaries = [
            self._run_task_with_limit(summary) for summary in tasks
        ]
        summary_chunks = await asyncio.gather(*summaries)

        text_summary = "\n".join(chunk for chunk in summary_chunks if valid_chunk(chunk))
        total_tokens = ApiBase.count_tokens(
            text_summary,
            model=self.llm_caller.kwargs.get("model", "gpt-4"),
        )
        self.logger.debug(f"Final summary contains {total_tokens} tokens")
        return text_summary


    async def check_for_restrictions(self, text_chunks: list[str]) -> str:
        """Extract restriction ordinance text from input text chunks.

        Args:
            text_chunks (list[str]): List of strings, each of which represent a chunk of text.
                The order of the strings should be the order of the text chunks.

        Returns:
            str: Ordinance text extracted from text chunks.
        """
        return await self._process(
            text_chunks=text_chunks,
            instructions=self._MODEL_INSTRUCTIONS_RESTRICTIONS,
            valid_chunk=_valid_chunk_not_short,
        )

    async def check_for_correct_size(self, text_chunks: list[str]) -> str:
        """Extract ordinance text from input text chunks for large WES.

        Args:
            text_chunks (list[str]): List of strings, each of which represent a chunk of text.
                The order of the strings should be the order of the text chunks.

        Returns:
            str: Ordinance text extracted from text chunks.
        """
        return await self._process(
            text_chunks=text_chunks,
            instructions=self._MODEL_INSTRUCTIONS_SIZE,
            valid_chunk=_valid_chunk,
        )


def _valid_chunk(chunk: str) -> bool:
    """True if chunk has content."""
    return True if chunk and "no relevant text" not in chunk.lower() else False


def _valid_chunk_not_short(chunk: str) -> bool:
    """True if chunk has content and is not too short."""
    return _valid_chunk(chunk) and len(chunk) > 20
