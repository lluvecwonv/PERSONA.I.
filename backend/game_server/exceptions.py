class LangChainServiceError(Exception):
    """General LangChain service error."""
    pass


class APIKeyNotFoundError(LangChainServiceError):
    """API key not configured."""
    pass


class SessionNotFoundError(LangChainServiceError):
    """Session not found."""
    pass


class ModelNotSupportedError(LangChainServiceError):
    """Unsupported model."""
    pass


class ChainExecutionError(LangChainServiceError):
    """Error during chain execution."""
    pass
