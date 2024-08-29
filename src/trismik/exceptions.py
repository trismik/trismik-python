class TrismikError(Exception):
    """
    Base class for all exceptions raised.
    Raised when an error occurs in the Trismik package, i.e. configuration.
    """
    pass


class TrismikApiError(TrismikError):
    """
    Raised when an error occurs while interacting with the Trismik API
    """
    pass
