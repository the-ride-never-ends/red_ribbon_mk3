
# NOTE: See /home/kylerose1946/american_law_search/app/app.py for this file's usage.

from typing import Annotated as Ann, Any, Optional


from pydantic import (
    AfterValidator as AV, 
    BaseModel, 
    BeforeValidator as BV, 
    Field, 
    PositiveInt,
    PlainSerializer,
)


def _check_gnis(value: Any) -> int:
    if not isinstance(value, (str, int)):
        raise TypeError(f"GNIS must be an integer or string, got {type(value).__name__}")
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"GNIS must be convertible to an integer, got value: {value}") from e

GNIS = Ann[
    PositiveInt,
    BV(_check_gnis),
    PlainSerializer(return_type=int),
]

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
