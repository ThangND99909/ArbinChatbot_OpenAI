import unicodedata
import re

def remove_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", text).strip()

def normalize_vi_text(text: str) -> str:
    """Chuẩn hóa tiếng Việt để tìm kiếm/embedding."""
    return remove_accents(text.lower().strip())
