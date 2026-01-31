from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Quote:
    weight_kg: float
    zone: str
    cost: float


@dataclass
class ParseError:
    code: str
    detail: str


@dataclass
class Result:
    quote: Optional[Quote] = None
    error: Optional[ParseError] = None

    def ok(self) -> bool:
        return self.quote is not None and self.error is None


@dataclass
class Patch:
    patch_id: str
    trigger: str
    unit_conversions: Dict[str, float]
    dest_aliases: Dict[str, str]
    item_aliases: Dict[str, str]
    parcel_aliases: Dict[str, str]
    prohibited_items: List[str]
    hazmat_items: List[str]
    liquid_items: List[str]
    embargo_dests: List[str]
    parcel_max_kg: Dict[str, float]
    example_input: str
    example_output: str
    tests: List[Dict[str, object]]
    status: str
    trigger_embedding: List[float]

    def matches(self, request: str) -> bool:
        return self.trigger.lower() in request.lower()


@dataclass
class Trace:
    request: str
    result: Result
    carrier_feedback: "CarrierFeedback"
    ok: bool
    baseline_ok: bool
    applied_patch_ids: List[str]
    retrieved_patch_ids: List[str]
    failure_cluster: Optional[str] = None


@dataclass
class SkillConfig:
    unit_conversions: Dict[str, float]
    dest_aliases: Dict[str, str]
    item_aliases: Dict[str, str]
    parcel_aliases: Dict[str, str]
    base_fee: float
    per_kg_rate: Dict[str, float]
    prohibited_items: Set[str]
    hazmat_items: Set[str]
    liquid_items: Set[str]
    embargo_dests: Set[str]
    parcel_max_kg: Dict[str, float]
    liquid_allowed_parcels: Set[str]

    def clone(self) -> "SkillConfig":
        return SkillConfig(
            unit_conversions=dict(self.unit_conversions),
            dest_aliases=dict(self.dest_aliases),
            item_aliases=dict(self.item_aliases),
            parcel_aliases=dict(self.parcel_aliases),
            base_fee=self.base_fee,
            per_kg_rate=dict(self.per_kg_rate),
            prohibited_items=set(self.prohibited_items),
            hazmat_items=set(self.hazmat_items),
            liquid_items=set(self.liquid_items),
            embargo_dests=set(self.embargo_dests),
            parcel_max_kg=dict(self.parcel_max_kg),
            liquid_allowed_parcels=set(self.liquid_allowed_parcels),
        )


@dataclass
class CarrierFeedback:
    ok: bool
    error_code: Optional[str] = None
    error_context: Dict[str, object] = field(default_factory=dict)
    quote: Optional[Quote] = None


@dataclass
class GoldenEval:
    request: str
    true_label: str
    baseline_label: str
    patched_label: str


@dataclass
class Metrics:
    total: int = 0
    ok: int = 0
    baseline_ok: int = 0
    failures: Dict[str, int] = field(default_factory=dict)
    patches_created: int = 0
    patches_active: int = 0
    patches_quarantined: int = 0
    applied_patches: int = 0
    retrieved_patches: int = 0
    ok_window: List[int] = field(default_factory=list)

    def record_failure(self, code: str) -> None:
        self.failures[code] = self.failures.get(code, 0) + 1
