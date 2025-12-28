"""Configuration management for protocol query."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment."""

    # Database
    db_path: Path = field(default_factory=lambda: Path("./data/protocols.db"))

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM
    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Search
    default_search_mode: str = "hybrid"
    default_result_limit: int = 10
    rrf_k: int = 60

    @classmethod
    def load(cls, env_file: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        db_path_str = os.getenv("PROTOCOL_DB_PATH", "./data/protocols.db")

        return cls(
            db_path=Path(db_path_str),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        )

    def ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
