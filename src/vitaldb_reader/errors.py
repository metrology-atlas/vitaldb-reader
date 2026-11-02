"""Errors raised by the reader."""


class VitalFormatError(ValueError):
    """The file is not a readable ``.vital`` file (bad signature, truncated or malformed packets)."""


class VitalValidationError(ValueError):
    """An argument or setting the reader does not accept."""
