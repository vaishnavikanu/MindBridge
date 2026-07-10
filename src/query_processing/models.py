from dataclasses import dataclass, field
from typing import List


@dataclass
class ProcessedQuery:
    """
    Represents a query after preprocessing but before retrieval.
    This object is intentionally extensible for future GraphRAG
    and multilingual enhancements.
    """

    # User's original query
    original_query: str

    # Query that the retriever will use
    english_query: str

    # ISO Language Code
    language: str

    # Whether translation was performed
    translation_required: bool

    # Detector confidence
    confidence: float

    #
    # -------- Future GraphRAG Fields --------
    #

    # Medical entities extracted from the query
    entities: List[str] = field(default_factory=list)

    # Expanded graph entities
    expanded_entities: List[str] = field(default_factory=list)

    # Retrieved graph nodes
    graph_nodes: List[str] = field(default_factory=list)