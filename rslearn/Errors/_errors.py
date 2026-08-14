class LengthError(Exception):
    """Raised when an input or path has an unexpected length."""
    def __init__(self, *args):
        super().__init__(*args)
        pass

class NotFittedError(Exception):
    """Raised when the model or data has not been fitted yet."""
    def __init__(self, *args):
        super().__init__(*args)
        pass

class InvalidShape(Exception):
    """Raised when an array or input has an incorrect shape."""
    def __init__(self, *args):
        super().__init__(*args)
        pass

class InvalidValueError(Exception):
    """Raised when an input contains values outside the expected range or type."""
    def __init__(self, *args):
        super().__init__(*args)
        pass

class InternelError(Exception):
    """Raised for internal errors that are not handled elsewhere."""
    def __init__(self, *args):
        super().__init__(*args)
        pass

class Error(Exception):
    """Base custom exception used throughout the application."""
    def __init__(self, *args):
        super().__init__(*args)
        pass