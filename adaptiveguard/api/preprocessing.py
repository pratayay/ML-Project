"""Input preprocessing for obfuscation-resistant moderation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256

_HOMOGLYPH_MAP = str.maketrans(
    {
        "а": "a",  # Cyrillic a
        "е": "e",  # Cyrillic e
        "о": "o",  # Cyrillic o
        "р": "p",  # Cyrillic er
        "с": "c",  # Cyrillic es
        "х": "x",  # Cyrillic ha
        "і": "i",  # Cyrillic i
        "Α": "A",  # Greek Alpha
        "Β": "B",  # Greek Beta
        "Ε": "E",  # Greek Epsilon
        "Ι": "I",  # Greek Iota
        "Κ": "K",  # Greek Kappa
        "Μ": "M",  # Greek Mu
        "Ν": "N",  # Greek Nu
        "Ο": "O",  # Greek Omicron
        "Ρ": "P",  # Greek Rho
        "Τ": "T",  # Greek Tau
        "Χ": "X",  # Greek Chi
        "α": "a",  # Greek alpha
        "β": "b",  # Greek beta
        "γ": "y",  # Greek gamma fallback
        "ι": "i",  # Greek iota
        "κ": "k",  # Greek kappa
        "ο": "o",  # Greek omicron
        "ρ": "p",  # Greek rho
        "τ": "t",  # Greek tau
        "χ": "x",  # Greek chi
    }
)

_LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
}

_REPEAT_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessingResult:
    """Preprocessed text plus safe-to-log transformation metadata."""

    transformed_text: str
    unicode_normalized: bool
    homoglyph_replacements: int
    repeated_collapses: int
    leetspeak_decoded: int
    typo_density: float
    obfuscation_score: float
    anomaly_flags: list[str]
    diff_metadata: dict[str, str | int | float | bool]


def _collapse_repeated_characters(text: str) -> tuple[str, int]:
    total_collapses = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal total_collapses
        total_collapses += 1
        return match.group(1) * 2

    return _REPEAT_PATTERN.sub(_replace, text), total_collapses


def _decode_leetspeak(text: str) -> tuple[str, int]:
    decoded_chars = 0
    output: list[str] = []

    for char in text:
        replacement = _LEETSPEAK_MAP.get(char)
        if replacement is None:
            output.append(char)
            continue
        decoded_chars += 1
        output.append(replacement)

    return "".join(output), decoded_chars


def preprocess_text(text: str) -> PreprocessingResult:
    """Normalize text and derive obfuscation signals without retaining raw content in logs."""
    normalized = unicodedata.normalize("NFKC", text)
    unicode_normalized = normalized != text

    homoglyph_mapped = normalized.translate(_HOMOGLYPH_MAP)
    homoglyph_replacements = sum(1 for before, after in zip(normalized, homoglyph_mapped) if before != after)

    collapsed, repeated_collapses = _collapse_repeated_characters(homoglyph_mapped)
    decoded, leetspeak_decoded = _decode_leetspeak(collapsed)

    changed_chars = sum(1 for before, after in zip(text, decoded) if before != after)
    length_delta = len(decoded) - len(text)
    typo_density = round((repeated_collapses + leetspeak_decoded) / max(len(text), 1), 4)
    obfuscation_score = min(
        1.0,
        round((homoglyph_replacements * 0.35 + repeated_collapses * 0.25 + leetspeak_decoded * 0.25 + int(unicode_normalized) * 0.15), 4),
    )

    anomaly_flags: list[str] = []
    if unicode_normalized:
        anomaly_flags.append("unicode_variant")
    if homoglyph_replacements > 0:
        anomaly_flags.append("homoglyph_substitution")
    if repeated_collapses > 0:
        anomaly_flags.append("repeated_characters")
    if leetspeak_decoded > 0:
        anomaly_flags.append("leetspeak")

    diff_metadata: dict[str, str | int | float | bool] = {
        "before_sha256_8": sha256(text.encode("utf-8")).hexdigest()[:8],
        "after_sha256_8": sha256(decoded.encode("utf-8")).hexdigest()[:8],
        "before_length": len(text),
        "after_length": len(decoded),
        "char_changes": changed_chars,
        "length_delta": length_delta,
        "unicode_normalized": unicode_normalized,
        "homoglyph_replacements": homoglyph_replacements,
        "repeated_collapses": repeated_collapses,
        "leetspeak_decoded": leetspeak_decoded,
    }

    return PreprocessingResult(
        transformed_text=decoded,
        unicode_normalized=unicode_normalized,
        homoglyph_replacements=homoglyph_replacements,
        repeated_collapses=repeated_collapses,
        leetspeak_decoded=leetspeak_decoded,
        typo_density=typo_density,
        obfuscation_score=obfuscation_score,
        anomaly_flags=anomaly_flags,
        diff_metadata=diff_metadata,
    )
