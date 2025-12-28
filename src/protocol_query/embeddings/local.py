"""Local embeddings using sentence-transformers."""

from typing import Optional

from protocol_query.core.config import Config, get_config


class LocalEmbeddings:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.model_name = self.config.embedding_model
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.embedding_dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]
