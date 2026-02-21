import os

import faiss
import numpy as np

from sam3_fursearch.config import Config


class VectorIndex:
    def __init__(
        self,
        index_path: str = Config.INDEX_PATH,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hnsw_m: int = Config.HNSW_M,
        ef_construction: int = Config.HNSW_EF_CONSTRUCTION,
        ef_search: int = Config.HNSW_EF_SEARCH
    ):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = self._load_or_create_index()
        # Update embedding_dim from loaded index (may differ from constructor default)
        self.embedding_dim = self.index.d
        if embedding_dim != self.embedding_dim:
            print(f"WARN: Vector index {index_path} has {self.embedding_dim} dimensions, not the provided default {embedding_dim}")

    def _load_or_create_index(self) -> faiss.IndexHNSWFlat:
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def add(self, embeddings: np.ndarray) -> int:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype(np.float32)
        start_id = self.index.ntotal
        self.index.add(embeddings)
        return start_id

    def search(self, query: np.ndarray, top_k: int = Config.DEFAULT_TOP_K) -> tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        return self.index.search(query, top_k)

    def save(self, backup: bool = False):
        if backup and os.path.exists(self.index_path):
            backup_path = f"{self.index_path}.bak"
            try:
                os.replace(self.index_path, backup_path)
            except OSError:
                pass  # Backup failed, continue with save
        faiss.write_index(self.index, self.index_path)

    @property
    def size(self) -> int:
        return self.index.ntotal

    @property
    def index_type(self) -> str:
        return "hnsw"

    def reconstruct(self, idx: int) -> np.ndarray:
        return self.index.reconstruct(int(idx))

    def reset(self):
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
