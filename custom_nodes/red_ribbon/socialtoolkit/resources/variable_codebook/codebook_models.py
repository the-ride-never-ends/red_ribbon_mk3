"""
Dataclasses for VariableCodebook
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DataType(str, Enum):
    """Enumeration of supported data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"


@dataclass
class CategoryValue:
    """Represents a category value in a categorical variable"""
    code: str
    label: str
    description: Optional[str] = None
    order: Optional[int] = None
    is_default: bool = False


@dataclass
class Variable:
    """Represents a variable in the codebook"""
    id: str
    name: str
    description: Optional[str] = None
    data_type: DataType = DataType.STRING
    categories: Optional[List[CategoryValue]] = None
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[Any] = None
    format_pattern: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodebookGroup:
    """Represents a group of variables in the codebook"""
    id: str
    name: str
    description: Optional[str] = None
    variables: List[Variable] = field(default_factory=list)
    parent_group_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Codebook:
    """Represents a complete codebook"""
    id: str
    name: str
    description: Optional[str] = None
    version: str
    groups: List[CodebookGroup] = field(default_factory=list)
    variables: List[Variable] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)