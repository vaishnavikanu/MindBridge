from typing import List

from loguru import logger

from src.chunking.chunker import Chunk
from src.graph.entity_extractor import EntityExtractor
from src.graph.relation_extractor import RelationExtractor
from src.graph.graph_store import GraphStore


class GraphBuilder:

    def __init__(self, graph_store: GraphStore):

        self.graph_store = graph_store

        self.entity_extractor = EntityExtractor()

        self.relation_extractor = RelationExtractor()

    ####################################################################
    # Build Graph
    ####################################################################

    def build_from_chunks(
        self,
        chunks: List[Chunk],
    ):

        logger.info(
            f"Building graph from {len(chunks)} chunks..."
        )

        for chunk in chunks:

            self._process_chunk(chunk)

        self.graph_store.save()

        logger.info(
            f"Graph construction complete."
        )

    ####################################################################
    # Process One Chunk
    ####################################################################

    def _process_chunk(
        self,
        chunk: Chunk,
    ):

        entities = self.entity_extractor.extract(
            chunk.text
        )

        #
        # Remove duplicates inside one chunk
        #
        unique_entities = {}

        for entity in entities:

            unique_entities[
                entity["entity"]
            ] = entity

        entities = list(unique_entities.values())

        #
        # Add Nodes
        #

        for entity in entities:

            self.graph_store.add_node(

                node_id=entity["entity"],

                label=entity["entity"],

                entity_type=entity["type"],

                document_id=chunk.metadata.get(
                    "source"
                ),

                chunk_id=chunk.chunk_id,

            )

        #
        # Build Relations
        #

        relations = self.relation_extractor.extract(
            entities
        )

        seen_edges = set()

        for relation in relations:

            edge = (
                relation["source"],
                relation["target"],
                relation["relation"],
            )

            if edge in seen_edges:
                continue

            seen_edges.add(edge)

            self.graph_store.add_edge(

                source=relation["source"],

                target=relation["target"],

                relation=relation["relation"],

                document_id=chunk.metadata.get(
                    "source"
                ),

                chunk_id=chunk.chunk_id,

                weight=relation["weight"],

            )