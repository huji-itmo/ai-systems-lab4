import pandas as pd


def sanitize_id(text: str) -> str:
    """Convert a string into a valid Textual widget ID."""
    # Replace spaces and invalid chars with underscores
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical 'Extracurricular Activities' to binary."""
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    return df
