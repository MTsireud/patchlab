import hashlib
import re
from typing import Dict, List, Optional, Tuple

from .models import CarrierFeedback, ParseError, Quote, Result, SkillConfig

_NUMBER_UNIT_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>[a-zA-Z]+)")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

BASE_UNIT_CONVERSIONS = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "kgs": 1.0,
}

GLOBAL_UNIT_CONVERSIONS = {
    **BASE_UNIT_CONVERSIONS,
    "lb": 0.453592,
    "lbs": 0.453592,
    "pound": 0.453592,
    "pounds": 0.453592,
    "oz": 0.0283495,
    "ounce": 0.0283495,
    "ounces": 0.0283495,
}

BASE_DEST_ALIASES = {
    "us": "US",
    "eu": "EU",
    "apac": "APAC",
    "north korea": "APAC",
    "iran": "APAC",
    "syria": "EU",
}

GLOBAL_DEST_ALIASES = {
    **BASE_DEST_ALIASES,
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "uk": "EU",
    "united kingdom": "EU",
    "britain": "EU",
    "england": "EU",
    "europe": "EU",
    "european union": "EU",
    "asia": "APAC",
    "japan": "APAC",
    "jp": "APAC",
    "australia": "APAC",
    "aus": "APAC",
    "canada": "US",
}

BASE_ITEM_ALIASES = {
    "books": "books",
    "book": "books",
    "clothes": "clothes",
    "toys": "toys",
    "electronics": "electronics",
    "laptop": "electronics",
}

GLOBAL_ITEM_ALIASES = {
    **BASE_ITEM_ALIASES,
    "battery": "battery",
    "lithium battery": "battery",
    "paint": "paint",
    "acid": "acid",
    "perfume": "perfume",
    "alcohol": "alcohol",
    "knife": "knife",
    "weapon": "weapon",
    "fireworks": "fireworks",
}

BASE_PARCEL_ALIASES = {
    "box": "box",
    "letter": "letter",
    "envelope": "letter",
}

GLOBAL_PARCEL_ALIASES = {
    **BASE_PARCEL_ALIASES,
    "tube": "tube",
    "crate": "crate",
    "pallet": "pallet",
}

PROHIBITED_ITEMS = {"fireworks", "weapon", "knife"}
HAZMAT_ITEMS = {"battery", "paint", "acid"}
LIQUID_ITEMS = {"perfume", "alcohol"}
EMBARGO_DESTS = {"north korea", "iran", "syria"}

BASE_PARCEL_MAX_KG = {
    "box": 30.0,
    "letter": 1.0,
}

GLOBAL_PARCEL_MAX_KG = {
    "box": 20.0,
    "letter": 0.5,
    "tube": 5.0,
    "crate": 50.0,
    "pallet": 500.0,
}

LIQUID_ALLOWED_PARCELS = {"crate", "pallet"}

PER_KG_RATE = {
    "US": 6.0,
    "EU": 7.5,
    "APAC": 9.0,
}


def make_base_config() -> SkillConfig:
    return SkillConfig(
        unit_conversions=dict(BASE_UNIT_CONVERSIONS),
        dest_aliases=dict(BASE_DEST_ALIASES),
        item_aliases=dict(BASE_ITEM_ALIASES),
        parcel_aliases=dict(BASE_PARCEL_ALIASES),
        base_fee=5.0,
        per_kg_rate=dict(PER_KG_RATE),
        prohibited_items=set({"fireworks"}),
        hazmat_items=set(),
        liquid_items=set(),
        embargo_dests=set(),
        parcel_max_kg=dict(BASE_PARCEL_MAX_KG),
        liquid_allowed_parcels=set({"crate", "pallet"}),
    )


def make_carrier_config() -> SkillConfig:
    return SkillConfig(
        unit_conversions=dict(GLOBAL_UNIT_CONVERSIONS),
        dest_aliases=dict(GLOBAL_DEST_ALIASES),
        item_aliases=dict(GLOBAL_ITEM_ALIASES),
        parcel_aliases=dict(GLOBAL_PARCEL_ALIASES),
        base_fee=5.0,
        per_kg_rate=dict(PER_KG_RATE),
        prohibited_items=set(PROHIBITED_ITEMS),
        hazmat_items=set(HAZMAT_ITEMS),
        liquid_items=set(LIQUID_ITEMS),
        embargo_dests=set(EMBARGO_DESTS),
        parcel_max_kg=dict(GLOBAL_PARCEL_MAX_KG),
        liquid_allowed_parcels=set(LIQUID_ALLOWED_PARCELS),
    )


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _extract_weight(text: str) -> Tuple[Optional[float], Optional[str]]:
    match = _NUMBER_UNIT_RE.search(text)
    if not match:
        return None, None
    raw_value = match.group("value").replace(",", ".")
    try:
        value = float(raw_value)
    except ValueError:
        return None, None
    unit = match.group("unit").lower()
    return value, unit


def _find_phrase(text: str, aliases: Dict[str, str], max_len: int = 3) -> Optional[str]:
    tokens = _tokenize(text)
    for n in range(max_len, 0, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in aliases:
                return phrase
    return None


def _noise_flip(request: str, key: str, rate: float) -> bool:
    if rate <= 0:
        return False
    digest = hashlib.sha1(f"{key}:{request}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < rate


def extract_weight_unit(request: str) -> Tuple[Optional[float], Optional[str]]:
    return _extract_weight(request)


def find_destination(request: str, aliases: Dict[str, str]) -> Optional[str]:
    return _find_phrase(request, aliases, max_len=3)


def find_item(request: str, aliases: Dict[str, str]) -> Optional[str]:
    return _find_phrase(request, aliases, max_len=3)


def find_parcel(request: str, aliases: Dict[str, str]) -> Optional[str]:
    return _find_phrase(request, aliases, max_len=2)


def _evaluate_request(
    request: str,
    config: SkillConfig,
    noise: float = 0.0,
) -> Tuple[Result, List[str], Dict[str, object]]:
    steps: List[str] = []
    context: Dict[str, object] = {}

    weight, unit = _extract_weight(request)
    steps.append(f"extract weight/unit -> {weight} {unit}")
    context["weight"] = weight
    context["unit"] = unit

    if _noise_flip(request, "drop_unit", noise):
        steps.append("noise: drop unit")
        unit = None

    if weight is None or unit is None:
        steps.append("error: no_weight")
        return Result(error=ParseError(code="no_weight", detail="No weight/unit detected")), steps, context

    if unit not in config.unit_conversions:
        steps.append(f"error: unit_unknown ({unit})")
        return Result(error=ParseError(code="unit_unknown", detail=unit)), steps, context

    weight_kg = weight * config.unit_conversions[unit]
    steps.append(f"unit conversion -> {config.unit_conversions[unit]}")
    context["weight_kg"] = weight_kg

    dest_phrase = _find_phrase(request, config.dest_aliases, max_len=3)
    if _noise_flip(request, "drop_dest", noise):
        steps.append("noise: drop dest")
        dest_phrase = None
    steps.append(f"find destination -> {dest_phrase}")
    if dest_phrase is None:
        steps.append("error: dest_unknown")
        return Result(error=ParseError(code="dest_unknown", detail="unknown")), steps, context

    zone = config.dest_aliases[dest_phrase]
    context["dest_phrase"] = dest_phrase
    context["zone"] = zone

    item_phrase = _find_phrase(request, config.item_aliases, max_len=3)
    if _noise_flip(request, "drop_item", noise):
        steps.append("noise: drop item")
        item_phrase = None
    steps.append(f"find item -> {item_phrase}")
    if item_phrase is None:
        steps.append("error: item_unknown")
        return Result(error=ParseError(code="item_unknown", detail="unknown")), steps, context

    item = config.item_aliases.get(item_phrase)
    context["item"] = item

    parcel_phrase = _find_phrase(request, config.parcel_aliases, max_len=2)
    if _noise_flip(request, "drop_parcel", noise):
        steps.append("noise: drop parcel")
        parcel_phrase = None
    steps.append(f"find parcel -> {parcel_phrase}")
    if parcel_phrase is None:
        steps.append("error: parcel_unknown")
        return Result(error=ParseError(code="parcel_unknown", detail="unknown")), steps, context

    parcel = config.parcel_aliases.get(parcel_phrase)
    context["parcel"] = parcel

    if _noise_flip(request, "ignore_embargo", noise):
        steps.append("noise: ignore embargo")
    elif dest_phrase in config.embargo_dests:
        steps.append(f"error: embargo_dest ({dest_phrase})")
        return Result(error=ParseError(code="embargo_dest", detail=dest_phrase)), steps, context

    if _noise_flip(request, "ignore_prohibited", noise):
        steps.append("noise: ignore prohibited")
    elif item in config.prohibited_items:
        steps.append(f"error: prohibited_item ({item})")
        return Result(error=ParseError(code="prohibited_item", detail=item)), steps, context

    if _noise_flip(request, "ignore_hazmat", noise):
        steps.append("noise: ignore hazmat")
    elif item in config.hazmat_items:
        steps.append(f"error: hazmat_item ({item})")
        return Result(error=ParseError(code="hazmat_item", detail=item)), steps, context

    if _noise_flip(request, "ignore_liquid", noise):
        steps.append("noise: ignore liquid rule")
    elif item in config.liquid_items and parcel not in config.liquid_allowed_parcels:
        steps.append(f"error: liquid_disallowed ({item})")
        return Result(error=ParseError(code="liquid_disallowed", detail=item)), steps, context

    max_kg = config.parcel_max_kg.get(parcel)
    if max_kg is not None and weight_kg > max_kg:
        steps.append(f"error: parcel_overweight ({parcel} max={max_kg})")
        return Result(error=ParseError(code="parcel_overweight", detail=parcel)), steps, context

    if zone not in config.per_kg_rate:
        steps.append(f"error: zone_unknown ({zone})")
        return Result(error=ParseError(code="zone_unknown", detail=zone)), steps, context

    cost = config.base_fee + config.per_kg_rate[zone] * weight_kg
    steps.append(f"zone -> {zone}")
    steps.append(f"weight_kg -> {round(weight_kg, 3)}")
    steps.append(f"cost -> {round(cost, 2)}")
    return Result(quote=Quote(weight_kg=round(weight_kg, 3), zone=zone, cost=round(cost, 2))), steps, context


def quote_from_request(request: str, config: SkillConfig, noise: float = 0.0) -> Result:
    result, _, _ = _evaluate_request(request, config, noise=noise)
    return result


def quote_from_request_trace(
    request: str, config: SkillConfig, noise: float = 0.0
) -> tuple[Result, List[str]]:
    result, steps, _ = _evaluate_request(request, config, noise=noise)
    return result, steps


def carrier_api_feedback(request: str) -> CarrierFeedback:
    config = make_carrier_config()
    result, _, context = _evaluate_request(request, config, noise=0.0)
    if result.ok():
        return CarrierFeedback(ok=True, quote=result.quote)

    error_context: Dict[str, object] = {}
    if result.error is not None:
        code = result.error.code
        if code in {"prohibited_item", "hazmat_item", "liquid_disallowed"}:
            error_context["item"] = context.get("item")
        if code == "embargo_dest":
            error_context["dest"] = context.get("dest_phrase")
        if code == "parcel_overweight":
            error_context["parcel"] = context.get("parcel")
            parcel = context.get("parcel")
            if parcel in config.parcel_max_kg:
                error_context["max_kg"] = config.parcel_max_kg[parcel]
        if code == "unit_unknown":
            error_context["unit"] = context.get("unit")
        if code == "dest_unknown":
            error_context["dest"] = context.get("dest_phrase")
        if code == "item_unknown":
            error_context["item"] = context.get("item")
        if code == "parcel_unknown":
            error_context["parcel"] = context.get("parcel")
        return CarrierFeedback(ok=False, error_code=code, error_context=error_context)

    return CarrierFeedback(ok=False, error_code="unknown", error_context={})
