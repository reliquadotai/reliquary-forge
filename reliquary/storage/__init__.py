__all__ = ["local_registry"]
from .local_registry import LocalArtifactRegistry
from .metadata_index import MetadataIndex, PostgresMetadataIndex, SqliteMetadataIndex
from .registry import ArtifactRegistry

try:
    from .object_registry import R2ArtifactRegistry
except ModuleNotFoundError:  # pragma: no cover - optional until object-store deps are installed
    R2ArtifactRegistry = None  # type: ignore[assignment]

__all__ = [
    "ArtifactRegistry",
    "LocalArtifactRegistry",
    "MetadataIndex",
    "PostgresMetadataIndex",
    "SqliteMetadataIndex",
    "R2ArtifactRegistry",
]
