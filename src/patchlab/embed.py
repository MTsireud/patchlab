import math
import re
from dataclasses import dataclass
from typing import List

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class Vector:
    values: List[float]

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))

    def dot(self, other: "Vector") -> float:
        return sum(a * b for a, b in zip(self.values, other.values))


class HashingEmbedder:
    def __init__(self, dims: int = 128) -> None:
        self.dims = dims

    def embed(self, text: str) -> Vector:
        vec = [0.0] * self.dims
        for token in _tokenize(text):
            idx = hash(token) % self.dims
            vec[idx] += 1.0
        return Vector(vec)


def cosine_similarity(a: Vector, b: Vector) -> float:
    denom = a.norm() * b.norm()
    if denom == 0.0:
        return 0.0
    return a.dot(b) / denom
