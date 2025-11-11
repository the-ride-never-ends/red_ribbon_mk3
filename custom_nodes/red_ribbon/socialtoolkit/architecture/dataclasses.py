
# NOTE: See /home/kylerose1946/american_law_search/app/app.py for this file's usage.

from functools import cached_property
from typing import Annotated as Ann, Any, Optional


from pydantic import (
    AfterValidator as AV, 
    BaseModel, 
    BeforeValidator as BV, 
    Field, 
    PositiveInt,
    NonNegativeInt,
    PlainSerializer,
    computed_field,
    ValidationError
)

from custom_nodes.red_ribbon.utils_.common import get_cid
from ._errors import DataclassError

def _check_gnis(value: Any) -> int:
    if not isinstance(value, (str, int)):
        raise TypeError(f"GNIS must be an integer or string, got {type(value).__name__}")
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"GNIS must be convertible to an integer, got value: {value}") from e

GNIS = Ann[PositiveInt, BV(_check_gnis)]

def _check_cid(value: Any) -> str:

    expected_prefix = "bafkrei"
    expected_length = 59

    match value:
        case str():
            if not value.startswith(expected_prefix):
                raise ValueError(f"CID must start with '{expected_prefix}', got '{value[:len(expected_prefix)]}'")
            if value != expected_length:
                raise ValueError(f"CID must be {expected_length} characters long, got {len(value)}")
            if not value.isalnum():
                raise ValueError("CID contains non-alphanumeric characters")
            if not value.islower():
                raise ValueError("CID contains upper case characters")
        case _:
            raise TypeError(f"CID must be a string, got {type(value).__name__}")


CID = Ann[str, AV(_check_cid)]


class HtmlParquet(BaseModel):
    cid: CID = Field(..., description="CID for the HTML content", examples=["bafkreigf4d7iyo2r4cfjohedcpriv22ina54t45o4jitjmcyhqmx4ptsge"])
    doc_id: CID = Field(..., description="Document ID", examples= ["CO_CH57RE", "CO_CHAPTERS_28_29RE"])
    doc_order: PositiveInt = Field(..., description="Order of the document in the collection")
    html_title : str = Field(..., description="Title of the HTML document")
    html_content: str = Field(..., description="HTML content of the document")
    gnis: GNIS = Field(..., description="Geographic Names Information System identifier. Unique at the municipal level.")

class CitationParquet(BaseModel):
    bluebook_cid: str = Field(..., description="Bluebook citation CID")
    cid: str = Field(..., description="CID for the HTML content")
    title: str = Field(..., description="Title of the cited document")
    public_law_number: str = Field(default="NA", description="Public law number. Used for federal laws. If not applicable, set to 'NA'.")
    chapter: str = Field(default="NA", description="Chapter title.")
    ordinance: Optional[str] = Field(default=None, description="Ordinance title.")
    section: Optional[str] = Field(default=None, description="Section number or identifier.")
    enacted: Optional[str] = Field(default=None, description="Enactment date of the law or regulation.")
    year: Optional[str] = Field(default=None, description="Year of enactment or publication.")
    history_note: str = Field(default="NA", description="History note or legislative history.")
    place_name: str = Field(..., description="Name of the place (e.g., city, county, state) associated with the citation.")
    state_code: str = Field(..., description="State code (e.g., 'CO' for Colorado) associated with the citation.")
    public_law_num: str = Field(default="NA", description="Public law number abbreviation.")
    bluebook_state_code: str = Field(..., description="Bluebook state code abbreviation (e.g., 'Ark.' for Arkansas).")
    state_name: str = Field(..., description="Full name of the state.")
    chapter_num: str = Field(..., description="Chapter number.")
    title_num: str = Field(..., description="Title number.")
    date: Optional[str] = Field(default=None, description="Date associated with the citation.")
    bluebook_citation: str = Field(..., description="Full Bluebook formatted citation.")
    gnis: GNIS = Field(..., description="Geographic Names Information System identifier. Unique at the municipal level.")


class EmbeddingParquet:
    embedding_cid: CID = Field(..., description="CID for the embedding")
    gnis: GNIS = Field(..., description="Geographic Names Information System identifier. Unique at the municipal level.")
    cid: CID = Field(..., description="CID for the HTML content")
    text_chunk_order: PositiveInt = Field(..., description="Order of the text chunk in the document")
    embedding: list[float] = Field(..., description="Embedding vector as a list of floats")


class Section(BaseModel):
    """
    Section from a document

    Attributes:
        cid (str): Law ID
        bluebook_cid (str): Bluebook citation ID
        title (str): Title of the law
        chapter (str): Chapter information
        place_name (str): Name of the place associated with the law
        state_name (str): Name of the state associated with the law
        bluebook_citation (str): Formatted Bluebook citation
        html (str, optional): HTML content if available
    """
    cid: str = Field(..., min_length=1)
    doc_order: NonNegativeInt
    bluebook_cid: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    chapter: str = Field(..., min_length=1)
    place_name: str = Field(..., min_length=1)
    state_name: str = Field(..., min_length=1)
    bluebook_citation: str = Field(..., min_length=1)
    html: Optional[str] = Field(default=None)


class Document(BaseModel):
    """
    Attributes:
        section_list (list[Section]): List of Section objects representing document content

    Properties:
        place_name (str): Name of the place associated with the document
        state_name (str): Name of the state associated with the document
        cid (str): Computed property that returns the document's cid
        content_json (list[dict[str, Any]]): Computed property that returns sorted rows as list of dicts
    """
    section_list: list[Section] = Field(exclude=True)

    _cid: str = None
    _place_name: str = None
    _state_name: str = None

    @computed_field # type: ignore[prop-decorator]
    @cached_property
    def cid(self) -> str:
        if self._cid is None and self.section_list:
            concatenated_cids = ''.join([section.cid for section in self.section_list])
            self._cid = get_cid(concatenated_cids)
        return self._cid

    @property
    def place_name(self) -> str:
        if self._place_name is None and self.section_list:
            self._place_name = self.section_list[0].place_name
        return self._place_name

    @property
    def state_name(self) -> str:
        if self._state_name is None and self.section_list:
            self._state_name = self.section_list[0].state_name
        return self._state_name

    @computed_field # type: ignore[prop-decorator]
    @cached_property
    def content_json(self) -> list[dict[str, Any]]:
        # Convert list of Section objects to list of dicts
        list_of_dicts = [section.model_dump() for section in self.section_list]

        # Sort the list of dicts by 'doc_order'
        sorted_list = sorted(list_of_dicts, key=lambda x: x['doc_order'])
        return sorted_list


class Vector(BaseModel):
    cid: str = Field(..., min_length=1)
    embedding: list[float]

def make_document(inputs: list[dict[str, Any]]) -> Document:
    """
    Create a Document object from a list of section dictionaries.

    Args:
        inputs (list[dict]): List of dictionaries, each representing a Section.

    Returns:
        Document: A Document object containing the provided sections.

    Raises:
        TypeError: If inputs is not a list or if any item in inputs is not a dict.
        ValueError: If inputs list is empty, or if any item in puts is an empty dict.
        DataclassError: If there is an error creating Section or Document objects.
    """
    if not isinstance(inputs, list):
        raise TypeError(f"inputs must be a list, got {type(inputs).__name__}")
    if not list:
        raise ValueError("inputs list cannot be empty")

    section_list = []
    for idx, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise TypeError(f"Each item in inputs must be a dict, got {type(item).__name__} at index {idx}")
        if not item:
            raise ValueError(f"Item at index {idx} is an empty dict")

        try:
            section = Section(**item)
        except ValidationError as e:
            raise DataclassError(f"Error creating Section object at index {idx}: {e}") from e
        section_list.append(section)

    try:
        document = Document(section_list=section_list)
    except ValidationError as e:
        raise DataclassError(f"Error creating Document object: {e}") from e

    return document