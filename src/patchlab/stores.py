from dataclasses import dataclass, field
from typing import List, Tuple

from .embed import Vector, cosine_similarity
from .models import Patch, Trace


@dataclass
class TraceStore:
    traces: List[Tuple[Trace, Vector]] = field(default_factory=list)

    def append(self, trace: Trace, embedding: Vector) -> None:
        self.traces.append((trace, embedding))

    def retrieve(self, query: Vector, k: int = 5) -> List[Trace]:
        scored = []
        for trace, emb in self.traces:
            scored.append((cosine_similarity(query, emb), trace))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]


@dataclass
class PatchStore:
    patches: List[Patch] = field(default_factory=list)

    def upsert(self, patch: Patch) -> None:
        for idx, existing in enumerate(self.patches):
            if existing.patch_id == patch.patch_id:
                self.patches[idx] = patch
                return
        self.patches.append(patch)

    def retrieve_active(self, query: Vector, k: int = 8) -> List[Patch]:
        scored = []
        for patch in self.patches:
            if patch.status != "active":
                continue
            emb = Vector(patch.trigger_embedding)
            scored.append((cosine_similarity(query, emb), patch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]

    def counts(self) -> Tuple[int, int]:
        active = sum(1 for p in self.patches if p.status == "active")
        quarantined = sum(1 for p in self.patches if p.status == "quarantined")
        return active, quarantined
