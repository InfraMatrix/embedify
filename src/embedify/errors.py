"""Core errors for embedify."""


class EmbedifyError(Exception):
    """Base error for embedify."""


class DetectionError(EmbedifyError):
    """Raised when a source cannot be detected."""


class LoaderError(EmbedifyError):
    """Raised when a loader fails on a source."""


class UnsupportedModalityError(EmbedifyError):
    """Raised when no loader/embedder supports a modality."""


class EmbedderError(EmbedifyError):
    """Raised when an embedder fails."""


class SinkError(EmbedifyError):
    """Raised when a sink fails."""


class ConfigError(EmbedifyError):
    """Raised for invalid configuration."""
