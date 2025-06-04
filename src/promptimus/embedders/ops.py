from .base import Embedding


def cosine(this: Embedding, other: Embedding) -> float:
    dot = sum(a * b for a, b in zip(this, other))
    norm_a = sum(a * a for a in this) ** 0.5
    norm_b = sum(b * b for b in other) ** 0.5
    return dot / (norm_a * norm_b)
