"""Embedding + Qdrant vector index.

Qdrant runs in `:memory:` mode -- the real client API, no Docker, no service to
start. One collection per chunking strategy so the two never share a search
space. Embeddings come from a local sentence-transformers model, so the whole
harness runs without an API key.
"""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

EMBED_MODEL = "all-MiniLM-L6-v2"     # 384-dim, ~90 MB, CPU-friendly


# BGE retrieval models are trained with an instruction on the query side only;
# the documentation is explicit that omitting it costs retrieval accuracy, and
# that passages must NOT carry it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    """Wraps the embedding model so it is loaded once and reused per strategy."""

    def __init__(self, model_name: str = EMBED_MODEL, query_prefix: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # renamed in sentence-transformers 5; the old name warns but still works
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_tokens = self.model.max_seq_length
        self.query_prefix = (
            BGE_QUERY_PREFIX if query_prefix is None and "bge" in model_name.lower()
            else (query_prefix or "")
        )

    def encode(self, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,     # cosine == dot on unit vectors
            show_progress_bar=progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([self.query_prefix + query])[0]

    def truncated_fraction(self, texts: list[str]) -> float:
        """Share of texts the model will silently cut off at max_seq_length.

        Worth reporting: it is not uniform across chunking strategies. Variable
        length clause chunks overrun a short context far more often than fixed
        windows do, which quietly penalises the strategy under test.
        """
        tok = self.model.tokenizer
        return sum(len(tok.encode(t)) > self.max_tokens for t in texts) / len(texts)


class ChunkIndex:
    """A Qdrant collection holding one chunking strategy's chunks."""

    def __init__(self, chunks: list[Chunk], encoder: Encoder, collection: str,
                 client: QdrantClient | None = None, progress: bool = False):
        self.collection = collection
        self.encoder = encoder
        self.chunks = chunks
        self.client = client or QdrantClient(":memory:")

        vectors = encoder.encode([c.text for c in chunks], progress=progress)
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=encoder.dim, distance=models.Distance.COSINE),
        )
        self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "start": c.start,
                        "end": c.end,
                        "strategy": c.strategy,
                    },
                )
                for i, (c, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

    def search(self, query: str, limit: int, doc_id: str | None = None) -> list[Chunk]:
        """Nearest chunks, optionally restricted to one document.

        CUAD asks its questions *of a given contract*, and the questions are
        templated per clause type -- "Which state/country's law governs the
        interpretation of the contract?" is worded identically for all 510
        contracts. Searching the whole corpus with such a query is unanswerable
        by construction: the Governing Law clause of every contract is an
        equally good match and nothing in the query says which one is meant.
        So retrieval is scoped with a payload filter, which is also how a real
        contract-review system narrows to the document under review.
        """
        vec = self.encoder.encode_query(query)
        flt = None
        if doc_id is not None:
            flt = models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=doc_id))])
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
        return [_chunk_from_payload(h.payload) for h in hits]


def _chunk_from_payload(p: dict) -> Chunk:
    return Chunk(
        doc_id=p["doc_id"],
        chunk_id=p["chunk_id"],
        text=p["text"],
        start=p["start"],
        end=p["end"],
        strategy=p["strategy"],
    )
