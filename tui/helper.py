import pandas as pd


def sanitize_id(text: str) -> str:
    """Convert a string into a valid Textual widget ID."""
    # Replace spaces and invalid chars with underscores
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized
