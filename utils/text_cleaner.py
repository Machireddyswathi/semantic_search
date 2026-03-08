# utils/text_cleaner.py
"""
Text preprocessing pipeline for the 20 Newsgroups dataset.

WHY A PIPELINE APPROACH:
Each cleaning step is a separate function. This makes it easy to:
- Unit test each step in isolation
- Skip steps for different datasets
- Add new steps without breaking existing ones

DESIGN PATTERN: Function composition — small, pure functions chained together.
"""

import re
import unicodedata
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def remove_email_headers(text: str) -> str:
    """
    Remove newsgroup/email headers like From:, Subject:, Lines:, etc.

    WHY: The 20 Newsgroups dataset is raw email data. Headers contain
    metadata noise (email addresses, dates, message IDs) that is not
    semantically meaningful for search. We strip everything before the
    first blank line (standard email structure: headers then blank line
    then body).
    """
    # Standard email/newsgroup format: headers end at first blank line
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1]
    return text


def remove_quoted_replies(text: str) -> str:
    """
    Remove quoted reply lines (lines starting with '>' or '|').

    WHY: Newsgroup threads quote previous messages. These quoted lines
    introduce duplicate/redundant content and pollute embeddings with
    someone else's words, not the author's original message.
    """
    lines = text.split("\n")
    cleaned = [line for line in lines if not line.strip().startswith(">")
               and not line.strip().startswith("|")]
    return "\n".join(cleaned)


def remove_urls(text: str) -> str:
    """
    Remove HTTP/FTP URLs.

    WHY: URLs are not semantically meaningful in isolation and often
    contain random hashes, session IDs, and paths that confuse embeddings.
    """
    url_pattern = re.compile(
        r"http[s]?://\S+|www\.\S+|ftp://\S+",
        re.IGNORECASE
    )
    return url_pattern.sub(" ", text)


def remove_email_addresses(text: str) -> str:
    """
    Remove email addresses.

    WHY: Email addresses (e.g., user@host.com) are identifiers, not
    meaningful semantic content. They fragment tokenization and add noise.
    """
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove non-alphanumeric characters except basic punctuation.

    WHY: Newsgroup posts contain MIME artifacts, encoding leftovers
    (=3D, =20), and ASCII art. We keep basic sentence punctuation
    because it helps sentence boundary detection in the embedder.
    """
    # Remove MIME encoding artifacts like =3D, =20
    text = re.sub(r"=[0-9A-Fa-f]{2}", " ", text)

    # Remove characters that are not letters, digits, or basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-]", " ", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/newlines into a single space.

    WHY: After all the removal steps above, the text is left with many
    gaps. We normalize so the embedding model sees clean token boundaries.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to ASCII equivalents where possible.

    WHY: Newsgroup posts can contain accented characters or special
    unicode that tokenizers handle inconsistently across platforms.
    """
    # NFC normalization: canonical decomposition then canonical composition
    text = unicodedata.normalize("NFC", text)

    # Encode to ASCII, ignoring characters that can't be converted
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str, min_length: int = 50) -> str | None:
    """
    Full cleaning pipeline. Returns None if document is too short after
    cleaning (not useful for semantic search).

    Args:
        text:       raw document string
        min_length: minimum character length to keep; shorter = discarded

    Returns:
        Cleaned string, or None if document should be discarded

    WHY min_length=50:
        Very short documents (< 50 chars) after cleaning are either
        empty threads, signature blocks, or fragments. They produce
        poor-quality embeddings and pollute the index.
    """
    if not text or not isinstance(text, str):
        return None

    # Apply each step in order — sequence matters!
    text = normalize_unicode(text)        # First: normalize encoding
    text = remove_email_headers(text)     # Then: strip metadata headers
    text = remove_quoted_replies(text)    # Then: remove quoted content
    text = remove_urls(text)              # Then: remove URLs
    text = remove_email_addresses(text)   # Then: remove email addresses
    text = remove_special_characters(text) # Then: remove non-semantic chars
    text = normalize_whitespace(text)     # Last: tidy up spacing

    # Discard documents that are too short after cleaning
    if len(text) < min_length:
        return None

    return text


def clean_batch(texts: list[str], min_length: int = 50) -> list[dict]:
    """
    Clean a list of documents and return structured records.

    Args:
        texts:      list of raw document strings
        min_length: passed through to clean_text

    Returns:
        List of dicts with keys: 'original_index', 'text'
        Only includes documents that passed the min_length filter.
    """
    results = []
    discarded = 0

    for idx, raw_text in enumerate(texts):
        cleaned = clean_text(raw_text, min_length=min_length)
        if cleaned is not None:
            results.append({
                "original_index": idx,
                "text": cleaned
            })
        else:
            discarded += 1

    logger.info(
        f"Cleaned {len(texts)} documents → "
        f"{len(results)} kept, {discarded} discarded"
    )
    return results