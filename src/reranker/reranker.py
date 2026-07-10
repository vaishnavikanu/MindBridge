import torch
import numpy as np
from typing import List, Tuple
from loguru import logger

from src.utils.config import config
from src.retrieval.retriever import RetrievalResult
from src.vectorstore.store import VectorRecord

#
# Global reranker cache
#
_RERANKER_CACHE = {}

class CrossEncoderReranker:
    def __init__(self, model_name: str = None, device: str = None, max_length: int = None):
        self.model_name = model_name or config.get("models.reranker.name", "BAAI/bge-reranker-base")
        self.device = ( device or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.max_length = max_length or config.get("models.reranker.max_length", 512)
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading reranker model: {self.model_name}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("Reranker model loaded successfully")

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
        if not results:
            return []

        top_k = top_k or len(results)
        pairs = [[query, result.chunk.text] for result in results]

        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            scores = self._model(**inputs).logits.squeeze(-1).cpu().numpy()

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored_results[:top_k]:
            result.score = float(score)
            reranked.append(result)

        logger.info(f"Reranked {len(results)} -> {len(reranked)} results")
        return reranked


class LightweightReranker:
    def __init__(self):
        self._embedders = {}

    def _get_embedder(self, embedder_type: str):
        if embedder_type not in self._embedders:
            from src.embeddings.embedder import get_embedder
            self._embedders[embedder_type] = get_embedder(embedder_type)
        return self._embedders[embedder_type]

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
        if not results:
            return []

        top_k = top_k or len(results)

        scored = []

        for result in results:

            chunk_emb = result.chunk.embedding

            if chunk_emb is None:
                continue

            embedder_type = "medcpt" if len(chunk_emb) == 768 else "sbert"
            query_emb = self._get_embedder(embedder_type).embed_query(query)

            similarity = float(np.inner(query_emb, chunk_emb))

            scored.append((result, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []

        for result, score in scored[:top_k]:

            result.score = float(score)

            reranked.append(result)

        logger.info(f"Lightweight reranked {len(results)} -> {len(reranked)} results")
        return reranked


def get_reranker(reranker_type: str = "cross_encoder"):

    global _RERANKER_CACHE

    if reranker_type not in _RERANKER_CACHE:

        if reranker_type == "cross_encoder":
            _RERANKER_CACHE[reranker_type] = CrossEncoderReranker()

        elif reranker_type == "lightweight":
            _RERANKER_CACHE[reranker_type] = LightweightReranker()

        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    return _RERANKER_CACHE[reranker_type]