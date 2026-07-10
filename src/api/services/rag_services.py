class RAGService:

    def __init__(self):

        self.pipeline = None

    def initialize(self):

        self.pipeline = RAGPipeline()

        self.pipeline.build_indexes()

    def query(...):

        return self.pipeline.query(...)