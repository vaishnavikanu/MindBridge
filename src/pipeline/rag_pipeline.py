import time
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from src.utils.config import config
from src.vectorstore.store import VectorStoreManager
from src.retrieval.retriever import PatientRetriever, ClinicianRetriever, RetrievalResult
from src.reranker.reranker import get_reranker
from src.generation.generator import get_generator
from src.chunking.chunker import ParentChildChunker
from src.embeddings.embedder import get_embedder
from src.ingestion.curated_kb import CuratedKBIngestion
from src.ingestion.user_content import UserContentIngestion
from src.query_processing.processor import QueryProcessor
from src.graph.graph_manager import GraphManager
from src.graph.graph_builder import GraphBuilder
from src.graph.graph_retriever import GraphRetriever



@dataclass
class PipelineResult:
    response: str
    retrieved_chunks: List[RetrievalResult]
    latency_ms: float
    role: str
    user_id: str

    prompt: Optional[str] = None
    retrieved_candidates: Optional[List[RetrievalResult]] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    retrieved_texts: Optional[List[str]] = None

class RAGPipeline:
    def __init__(
        self, 
        generator_type: str = "ollama",
        use_graph=True,
        use_cross_encoder=True,
        use_dual_embeddings=True,
        ):

        # self.vector_store_manager = VectorStoreManager()
        # self.patient_retriever = PatientRetriever(self.vector_store_manager)
        # self.clinician_retriever = ClinicianRetriever(self.vector_store_manager)
        #self.reranker = get_reranker(
        #   config.get("reranker.type", "lightweight")
        #)
        #self.generator = get_generator("mock" if use_mock_generator else "local")
        self.generator = get_generator(generator_type)
        self.query_processor = QueryProcessor()
        self.graph_manager = GraphManager()
        self.graph_retriever = GraphRetriever()       
        self.chunker = ParentChildChunker()
        self.sbert_embedder = get_embedder("sbert")
        self.medcpt_embedder = get_embedder("medcpt")
        self._indexes_built = False

        self.use_graph = use_graph

        self.use_cross_encoder = use_cross_encoder
        self.use_dual_embeddings = use_dual_embeddings

        logger.info("=" * 60)
        logger.info(f"Graph Enabled       : {self.use_graph}")
        logger.info(f"Dual Embeddings     : {self.use_dual_embeddings}")
        logger.info(f"Cross Encoder       : {self.use_cross_encoder}")
        logger.info("=" * 60)

        self.vector_store_manager = VectorStoreManager()

        self.patient_retriever = PatientRetriever(
            self.vector_store_manager,
            use_dual_embeddings=use_dual_embeddings,
        )

        self.clinician_retriever = ClinicianRetriever(
            self.vector_store_manager,
            use_dual_embeddings=use_dual_embeddings,
            use_graph=use_graph,
        )
        reranker = (
            "cross_encoder"
            if use_cross_encoder
            else "lightweight"
        )
        self.reranker = get_reranker(reranker)
        logger.info(f"Using reranker: {self.reranker.__class__.__name__}")

    def initialize(self):
        """
        Initialize the complete RAG pipeline.

        This method should be called exactly once when the
        application starts.
        """

        logger.info("Initializing RAG pipeline...")

        self.build_indexes()

        #
        # Demo data
        # (Later this will become dynamic ingestion)
        #
        # self.ingest_user_data("patient_001")
        # self.ingest_clinician_data("clinician_001")

        logger.info("Pipeline initialization complete.")

    @property
    def ready(self) -> bool:
        return self._indexes_built

    def build_indexes(self, force_rebuild: bool = False):
        if self._indexes_built and not force_rebuild:
            logger.info("Indexes already built, skipping")
            return

        logger.info("Building all indexes...")
        start_time = time.perf_counter()

        #
        # Skip rebuilding if FAISS indexes already exist
        #
        curated_sbert_store = self.vector_store_manager.get_store(
            "curated_kb_sbert",
            dimension=384,
        )

        curated_medcpt_store = self.vector_store_manager.get_store(
            "curated_kb_medcpt",
            dimension=768,
        )

        if (
            curated_sbert_store.index.ntotal > 0
            and
            curated_medcpt_store.index.ntotal > 0
            and
            not force_rebuild
        ):
            logger.info("Existing FAISS indexes found. Skipping index construction.")
            self._indexes_built = True
            return

        curated_ingestion = CuratedKBIngestion()
        curated_docs = curated_ingestion.ingest()

        if curated_docs:
            curated_chunks = self.chunker.chunk_documents(curated_docs)

            if self.use_graph:

                global_builder = GraphBuilder(
                    self.graph_manager.get_global_graph()
                )

                global_builder.build_from_chunks(
                    curated_chunks
                )

            # Split chunks into SBERT vs MedCPT
            sbert_chunks = []
            medcpt_chunks = []

            for c in curated_chunks:
                if c.metadata.get("chunk_type") != "child":
                    continue

                source = c.metadata.get("source", "").lower()

                if any(x in source for x in ["cbt", "dbt", "self_help"]):
                    sbert_chunks.append(c)
                else:
                    medcpt_chunks.append(c)

            # SBERT embeddings
            if sbert_chunks:
                sbert_embeddings = self.sbert_embedder.embed([c.text for c in sbert_chunks])
                # curated_sbert_store = self.vector_store_manager.get_store(
                #     "curated_kb_sbert",
                #     dimension=384
                # )
                curated_sbert_store.add_chunks(sbert_chunks, sbert_embeddings)

            # MedCPT embeddings

            logger.info("Starting MedCPT embedding...")
            
            if medcpt_chunks:
                medcpt_embeddings = self.medcpt_embedder.embed([c.text for c in medcpt_chunks])
                # curated_medcpt_store = self.vector_store_manager.get_store(
                #     "curated_kb_medcpt",
                #     dimension=768
                # )
                curated_medcpt_store.add_chunks(medcpt_chunks, medcpt_embeddings)

        self._indexes_built = True
        logger.info(f"Index building complete in {time.perf_counter() - start_time:.2f}s")

    def ingest_user_data(self, user_id: str):
        logger.info(f"Ingesting data for user: {user_id}")
        ingestion = UserContentIngestion()
        docs = ingestion.ingest_user_data(user_id)

        if docs:
            chunks = self.chunker.chunk_documents(docs)
            
            patient_builder = GraphBuilder(
                self.graph_manager.get_patient_graph(user_id)
            )

            patient_builder.build_from_chunks(
                chunks
            )
            
            child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
            embeddings = self.sbert_embedder.embed([c.text for c in child_chunks])
            store = self.vector_store_manager.get_store(
                f"user_{user_id}_private",
                dimension=384
            )
            store.add_chunks(child_chunks, embeddings)
            self.patient_retriever._bm25_built = False

    def ingest_clinician_data(self, clinician_id: str):
        logger.info(f"Ingesting data for clinician: {clinician_id}")
        ingestion = UserContentIngestion()
        docs = ingestion.ingest_clinician_data(clinician_id)

        if docs:
            chunks = self.chunker.chunk_documents(docs)
            
            clinician_builder = GraphBuilder(
                self.graph_manager.get_clinician_graph(
                    clinician_id
                )
            )

            clinician_builder.build_from_chunks(
                chunks
            )
            
            child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
            embeddings = self.medcpt_embedder.embed([c.text for c in child_chunks])
            store = self.vector_store_manager.get_store(
                f"clinician_{clinician_id}_private",
                dimension=768
            )
            store.add_chunks(child_chunks, embeddings)

    def query(self, query: str, role: str, user_id: str, top_k: int = None) -> PipelineResult:
        start_time = time.perf_counter()
        top_k = top_k or config.get("retrieval.final_top_k", 5)

        logger.info(
            f"Processing query: role={role}, user_id={user_id}, query={query[:50]}..."
        )

        #
        # Process the query (language detection, translation, etc.)
        #
        processed_query = self.query_processor.process(query)

        

        #
        # Optional graph expansion (for ablation)
        #
        expanded_entities = []

        if self.use_graph:

            graphs = self.graph_manager.get_graphs(
                role,
                user_id,
            )

            expanded_entities = self.graph_retriever.expand_query(
                processed_query.english_query,
                graphs,
            )

        else:

            expanded_entities = []

        expanded_query = processed_query.english_query

        if expanded_entities:
            expanded_terms = [
                term
                for term in expanded_entities
                if term.lower() != processed_query.english_query.lower()
            ]

            expanded_query = (
                processed_query.english_query
                + " "
                + " ".join(expanded_terms)
            )

        # logger.info(
        #     f"Retrieved using graph = {self.use_graph}" 
        # )

        if role == "patient":
            retrieved = self.patient_retriever.retrieve(
                expanded_query,
                user_id,
                top_k * 2,
            )   
        elif role == "clinician":
            retrieved = self.clinician_retriever.retrieve(
                expanded_query,
                user_id,
                top_k * 2,
            )

        else:
            raise ValueError(f"Unknown role: {role}")

        reranked = self.reranker.rerank(
            expanded_query,
            retrieved,
            top_k,
        )

        response = self.generator.generate_with_context(
            processed_query.english_query,
            reranked,
            role,
        )

        #
        # Translate response back to original language
        #
        response = self.query_processor.restore_response(
            response,
            processed_query,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            response=response,
            prompt=getattr(self.generator, "last_prompt", None),
            retrieved_chunks=reranked,
            retrieved_candidates=retrieved,
            latency_ms=latency_ms,
            role=role,
            user_id=user_id,

            retrieved_texts=[
                x.chunk.text
                for x in reranked
            ],
        )

    def get_stats(self) -> dict:

        return {
            "pipeline_ready": self.ready,
            "generator": self.generator.__class__.__name__,
            "reranker": self.reranker.__class__.__name__,
            "indexes_built": self._indexes_built,
            "stores": {
                name: store.get_stats()
                for name, store in self.vector_store_manager.stores.items()
            },
        }