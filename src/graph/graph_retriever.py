from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_expander import GraphExpander


class GraphRetriever:

    def __init__(self):

        self.entity_extractor = EntityExtractor()

        self.expander = GraphExpander()

    def expand_query(
        self,
        query,
        graphs,
    ):

        entities = self.entity_extractor.extract(query)

        entity_names = [
            e["entity"]
            for e in entities
        ]

        expanded = self.expander.expand(
            graphs,
            entity_names,
        )

        return expanded