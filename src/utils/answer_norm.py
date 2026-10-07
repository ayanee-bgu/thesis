"""
Lightweight answer normalization for MATH benchmark.
Extracts and normalizes answers for string-level comparison.
Does not require sympy — good enough for Level 1-2 problems.
"""

import re


def extract_boxed(text: str) -> str:
    """Pull the content of \\boxed{...} from a LaTeX string."""
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return match.group(1).strip() if match else text.strip()


def normalize(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    Handles common LaTeX, units, whitespace, and casing differences.
    """
    s = answer.strip()

    # Extract boxed if present
    s = extract_boxed(s)

    # Strip surrounding dollar signs
    s = re.sub(r"^\$+|^\$+", "", s).strip()
    s = s.strip("$")

    # Remove LaTeX formatting that doesn't affect value
    s = s.replace("\\,", " ").replace("\\ ", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Normalize fractions: \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)

    # Remove trailing units (km/h, km, m, etc.) — keep pure number
    s = re.sub(r"\s+(km/h|km|m|cm|mph|hours?|minutes?|seconds?|degrees?|°)$", "", s, flags=re.IGNORECASE)

    # Collapse whitespace, lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()

    # Remove commas in large numbers: 1,000 → 1000
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)

    return s


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Return True if predicted and ground_truth normalize to the same string."""
    return normalize(predicted) == normalize(ground_truth)
