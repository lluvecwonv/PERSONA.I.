class LangChainServiceError(Exception):
    """LangChain 서비스 관련 일반 예외"""
    pass


class APIKeyNotFoundError(LangChainServiceError):
    """API 키가 설정되지 않았을 때 발생하는 예외"""
    pass


class SessionNotFoundError(LangChainServiceError):
    """세션을 찾을 수 없을 때 발생하는 예외"""
    pass


class ModelNotSupportedError(LangChainServiceError):
    """지원하지 않는 모델일 때 발생하는 예외"""
    pass


class ChainExecutionError(LangChainServiceError):
    """체인 실행 중 오류가 발생했을 때"""
    pass
