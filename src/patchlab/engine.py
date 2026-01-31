import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .embed import HashingEmbedder
from .models import CarrierFeedback, GoldenEval, Metrics, Patch, Result, SkillConfig, Trace
from .stores import PatchStore, TraceStore
from .toy_app import (
    GLOBAL_DEST_ALIASES,
    GLOBAL_ITEM_ALIASES,
    GLOBAL_PARCEL_ALIASES,
    GLOBAL_UNIT_CONVERSIONS,
    carrier_api_feedback,
    find_destination,
    find_item,
    find_parcel,
    make_base_config,
    quote_from_request,
    quote_from_request_trace,
)


@dataclass
class SimulationConfig:
    runs: int = 500
    seed: int = 42
    show_patches: int = 5
    verbose: bool = False
    trace_runs: int = 5
    model_noise: float = 0.03
    golden_size: int = 40


REGRESSION_REQUESTS = [
    "Ship 2 kg books box to US",
    "Quote 1.5kg clothes box to EU",
    "Send 3 kgs toys box to APAC",
]


TEMPLATES = [
    "Ship {weight} {unit} {item} in a {parcel} to {dest}",
    "Quote for {weight}{unit} {item} {parcel} -> {dest}",
    "Need shipping {weight} {unit} {item} via {parcel} to {dest}",
    "Send {item} {weight}{unit} {parcel} {dest}",
]

BASE_UNITS = ["kg", "kilogram", "kgs"]
EDGE_UNITS = ["lb", "lbs", "pound", "oz", "ounces"]

BASE_DESTS = ["US", "EU", "APAC", "iran", "north korea"]
EDGE_DESTS = ["usa", "united states", "uk", "britain", "europe", "asia", "japan", "jp", "aus", "canada"]

BASE_ITEMS = ["books", "clothes", "toys", "electronics", "laptop"]
EDGE_ITEMS = ["battery", "lithium battery", "perfume", "alcohol", "paint", "knife", "weapon", "fireworks"]

BASE_PARCELS = ["box", "letter", "envelope"]
EDGE_PARCELS = ["tube", "crate", "pallet"]

GOLDEN_REQUESTS = [
    "Ship 2 kg books box to US",
    "Ship 1 kg clothes box to EU",
    "Send 0.5 kg toys letter to APAC",
    "Ship 2 kg books box to atlantis",
    "Send 2 kg mystery box to US",
    "Quote 2 kg books satchel to EU",
    "Ship 1 stone books box to US",
    "Ship 2 kg weapon box to US",
    "Ship 2 kg fireworks crate to US",
    "Ship 2 kg battery box to US",
    "Ship 2 kg paint crate to EU",
    "Ship 2 kg perfume box to US",
    "Ship 2 kg alcohol box to APAC",
    "Ship 1 kg books box to iran",
    "Ship 1 kg books box to north korea",
    "Ship 1 kg books letter to EU",
    "Ship 5 kg books letter to EU",
    "Ship 10 kg books tube to EU",
    "Ship 2 kg books crate to US",
    "Ship 1 kg laptop box to canada",
]


def _hash_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def _generate_request(rng: random.Random) -> str:
    weight = rng.choice(["0.2", "0.5", "1", "1.5", "2", "2.5", "3", "4", "5", "10", "15", "25"])
    if rng.random() < 0.7:
        unit = rng.choice(BASE_UNITS)
    else:
        unit = rng.choice(EDGE_UNITS)

    if rng.random() < 0.7:
        dest = rng.choice(BASE_DESTS)
    else:
        dest = rng.choice(EDGE_DESTS)

    if rng.random() < 0.7:
        item = rng.choice(BASE_ITEMS)
    else:
        item = rng.choice(EDGE_ITEMS)

    if rng.random() < 0.7:
        parcel = rng.choice(BASE_PARCELS)
    else:
        parcel = rng.choice(EDGE_PARCELS)

    template = rng.choice(TEMPLATES)
    return template.format(weight=weight, unit=unit, dest=dest, item=item, parcel=parcel)


def _apply_patches(config: SkillConfig, patches: List[Patch]) -> List[str]:
    applied = []
    for patch in patches:
        if patch.unit_conversions:
            config.unit_conversions.update(patch.unit_conversions)
        if patch.dest_aliases:
            config.dest_aliases.update(patch.dest_aliases)
        if patch.item_aliases:
            config.item_aliases.update(patch.item_aliases)
        if patch.parcel_aliases:
            config.parcel_aliases.update(patch.parcel_aliases)
        if patch.prohibited_items:
            config.prohibited_items.update(patch.prohibited_items)
        if patch.hazmat_items:
            config.hazmat_items.update(patch.hazmat_items)
        if patch.liquid_items:
            config.liquid_items.update(patch.liquid_items)
        if patch.embargo_dests:
            config.embargo_dests.update(patch.embargo_dests)
        if patch.parcel_max_kg:
            config.parcel_max_kg.update(patch.parcel_max_kg)
        applied.append(patch.patch_id)
    return applied


def _label_from_result(result: Result) -> str:
    return "ok" if result.ok() else result.error.code


def _label_from_feedback(feedback: CarrierFeedback) -> str:
    return "ok" if feedback.ok else (feedback.error_code or "unknown")


def _build_patch_from_feedback(
    request: str,
    predicted: Result,
    carrier: CarrierFeedback,
) -> Patch:
    patch_id_seed = None
    unit_conversions: Dict[str, float] = {}
    dest_aliases: Dict[str, str] = {}
    item_aliases: Dict[str, str] = {}
    parcel_aliases: Dict[str, str] = {}
    prohibited_items: List[str] = []
    hazmat_items: List[str] = []
    liquid_items: List[str] = []
    embargo_dests: List[str] = []
    parcel_max_kg: Dict[str, float] = {}

    if carrier.ok:
        if predicted.error is None:
            raise ValueError("No failure to patch")
        code = predicted.error.code
        if code == "unit_unknown":
            unit = predicted.error.detail
            if unit not in GLOBAL_UNIT_CONVERSIONS:
                raise ValueError("Unknown unit")
            unit_conversions[unit] = GLOBAL_UNIT_CONVERSIONS[unit]
            patch_id_seed = f"unit:{unit}"
        elif code == "dest_unknown":
            dest_phrase = find_destination(request, GLOBAL_DEST_ALIASES)
            if dest_phrase is None:
                raise ValueError("Unknown destination")
            dest_aliases[dest_phrase] = GLOBAL_DEST_ALIASES[dest_phrase]
            patch_id_seed = f"dest:{dest_phrase}"
        elif code == "item_unknown":
            item_phrase = find_item(request, GLOBAL_ITEM_ALIASES)
            if item_phrase is None:
                raise ValueError("Unknown item")
            item_aliases[item_phrase] = GLOBAL_ITEM_ALIASES[item_phrase]
            patch_id_seed = f"item:{item_phrase}"
        elif code == "parcel_unknown":
            parcel_phrase = find_parcel(request, GLOBAL_PARCEL_ALIASES)
            if parcel_phrase is None:
                raise ValueError("Unknown parcel")
            parcel_aliases[parcel_phrase] = GLOBAL_PARCEL_ALIASES[parcel_phrase]
            patch_id_seed = f"parcel:{parcel_phrase}"
        else:
            raise ValueError(f"No patch strategy for {code}")
    else:
        code = carrier.error_code or "unknown"
        if code == "prohibited_item":
            item = carrier.error_context.get("item")
            if item is None:
                raise ValueError("Missing item")
            prohibited_items.append(str(item))
            item_aliases[str(item)] = str(item)
            patch_id_seed = f"prohibited:{item}"
        elif code == "hazmat_item":
            item = carrier.error_context.get("item")
            if item is None:
                raise ValueError("Missing item")
            hazmat_items.append(str(item))
            item_aliases[str(item)] = str(item)
            patch_id_seed = f"hazmat:{item}"
        elif code == "liquid_disallowed":
            item = carrier.error_context.get("item")
            if item is None:
                raise ValueError("Missing item")
            liquid_items.append(str(item))
            item_aliases[str(item)] = str(item)
            patch_id_seed = f"liquid:{item}"
        elif code == "embargo_dest":
            dest = carrier.error_context.get("dest")
            if dest is None:
                raise ValueError("Missing destination")
            embargo_dests.append(str(dest))
            dest_aliases[str(dest)] = GLOBAL_DEST_ALIASES.get(str(dest), "APAC")
            patch_id_seed = f"embargo:{dest}"
        elif code == "parcel_overweight":
            parcel = carrier.error_context.get("parcel")
            max_kg = carrier.error_context.get("max_kg")
            if parcel is None or max_kg is None:
                raise ValueError("Missing parcel max weight")
            parcel_aliases[str(parcel)] = str(parcel)
            parcel_max_kg[str(parcel)] = float(max_kg)
            patch_id_seed = f"parcel_max:{parcel}"
        else:
            raise ValueError(f"No patch strategy for {code}")

    if patch_id_seed is None:
        raise ValueError("No patch id")

    trigger = patch_id_seed.split(":", 1)[1]
    patch_id = _hash_id(patch_id_seed)
    tests = _build_patch_tests(
        request=request,
        unit_conversions=unit_conversions,
        dest_aliases=dest_aliases,
        item_aliases=item_aliases,
        parcel_aliases=parcel_aliases,
        prohibited_items=prohibited_items,
        hazmat_items=hazmat_items,
        liquid_items=liquid_items,
        embargo_dests=embargo_dests,
        parcel_max_kg=parcel_max_kg,
    )

    return Patch(
        patch_id=patch_id,
        trigger=trigger,
        unit_conversions=unit_conversions,
        dest_aliases=dest_aliases,
        item_aliases=item_aliases,
        parcel_aliases=parcel_aliases,
        prohibited_items=prohibited_items,
        hazmat_items=hazmat_items,
        liquid_items=liquid_items,
        embargo_dests=embargo_dests,
        parcel_max_kg=parcel_max_kg,
        example_input=request,
        example_output="",
        tests=tests,
        status="candidate",
        trigger_embedding=[],
    )


def _build_patch_tests(
    request: str,
    unit_conversions: Dict[str, float],
    dest_aliases: Dict[str, str],
    item_aliases: Dict[str, str],
    parcel_aliases: Dict[str, str],
    prohibited_items: List[str],
    hazmat_items: List[str],
    liquid_items: List[str],
    embargo_dests: List[str],
    parcel_max_kg: Dict[str, float],
) -> List[Dict[str, object]]:
    tests: List[Dict[str, object]] = []
    restricted_labels = {}
    for item in prohibited_items:
        restricted_labels[item] = "prohibited_item"
    for item in hazmat_items:
        restricted_labels[item] = "hazmat_item"
    for item in liquid_items:
        restricted_labels[item] = "liquid_disallowed"

    if unit_conversions:
        unit = next(iter(unit_conversions.keys()))
        tests.append({"request": f"Ship 1 {unit} books box to US", "label": "ok"})
    if dest_aliases:
        dest = next(iter(dest_aliases.keys()))
        label = "embargo_dest" if dest in embargo_dests else "ok"
        tests.append({"request": f"Ship 1 kg books box to {dest}", "label": label})
    if item_aliases:
        item = next(iter(item_aliases.keys()))
        label = restricted_labels.get(item, "ok")
        tests.append({"request": f"Ship 1 kg {item} box to US", "label": label})
    if prohibited_items:
        item = prohibited_items[0]
        tests.append({"request": f"Ship 1 kg {item} box to US", "label": "prohibited_item"})
    if hazmat_items:
        item = hazmat_items[0]
        tests.append({"request": f"Ship 1 kg {item} box to US", "label": "hazmat_item"})
    if liquid_items:
        item = liquid_items[0]
        tests.append({"request": f"Ship 1 kg {item} box to US", "label": "liquid_disallowed"})
    if embargo_dests:
        dest = embargo_dests[0]
        tests.append({"request": f"Ship 1 kg books box to {dest}", "label": "embargo_dest"})
    if parcel_aliases:
        parcel = next(iter(parcel_aliases.keys()))
        safe_weight = 1.0
        if parcel in parcel_max_kg:
            safe_weight = max(parcel_max_kg[parcel] * 0.5, 0.1)
        tests.append({"request": f"Ship {safe_weight} kg books {parcel} to US", "label": "ok"})
    if parcel_max_kg:
        parcel = next(iter(parcel_max_kg.keys()))
        max_kg = parcel_max_kg[parcel]
        tests.append({"request": f"Ship {max_kg + 1} kg books {parcel} to US", "label": "parcel_overweight"})

    if not tests:
        tests.append({"request": request, "label": _label_from_feedback(carrier_api_feedback(request))})
    return tests


def _run_tests(config: SkillConfig, tests: List[Dict[str, object]]) -> bool:
    for test in tests:
        result = quote_from_request(test["request"], config, noise=0.0)
        label = _label_from_result(result)
        if label != test["label"]:
            return False
    return True


def _build_regression_suite() -> List[Dict[str, object]]:
    suite = []
    for request in REGRESSION_REQUESTS:
        label = _label_from_feedback(carrier_api_feedback(request))
        suite.append({"request": request, "label": label})
    return suite


def _evaluate_golden_set(
    sim: SimulationConfig,
    embedder: HashingEmbedder,
    patch_store: PatchStore,
) -> List[GoldenEval]:
    evals: List[GoldenEval] = []
    for request in GOLDEN_REQUESTS[: sim.golden_size]:
        carrier = carrier_api_feedback(request)
        true_label = _label_from_feedback(carrier)

        base_config = make_base_config()
        baseline_result = quote_from_request(request, base_config, noise=sim.model_noise)
        baseline_label = _label_from_result(baseline_result)

        patched_config = base_config.clone()
        query_vec = embedder.embed(request)
        patches = patch_store.retrieve_active(query_vec, k=8)
        for patch in patches:
            if patch.matches(request):
                _apply_patches(patched_config, [patch])

        patched_result = quote_from_request(request, patched_config, noise=sim.model_noise)
        patched_label = _label_from_result(patched_result)

        evals.append(
            GoldenEval(
                request=request,
                true_label=true_label,
                baseline_label=baseline_label,
                patched_label=patched_label,
            )
        )
    return evals


def run_simulation(sim: SimulationConfig) -> Tuple[Metrics, List[Trace], PatchStore, List[GoldenEval]]:
    rng = random.Random(sim.seed)
    embedder = HashingEmbedder()
    trace_store = TraceStore()
    patch_store = PatchStore()
    metrics = Metrics()
    regression_suite = _build_regression_suite()

    printed_traces = 0

    for run_idx in range(1, sim.runs + 1):
        request = _generate_request(rng)
        query_vec = embedder.embed(request)

        trace_store.retrieve(query_vec, k=5)
        active_patches = patch_store.retrieve_active(query_vec, k=8)

        base_config = make_base_config()
        patched_config = base_config.clone()

        retrieved_patch_ids = [p.patch_id for p in active_patches]
        applied_patch_ids = []
        for patch in active_patches:
            if patch.matches(request):
                applied_patch_ids.extend(_apply_patches(patched_config, [patch]))

        if sim.verbose:
            result, patched_steps = quote_from_request_trace(request, patched_config, noise=sim.model_noise)
            baseline, baseline_steps = quote_from_request_trace(request, base_config, noise=sim.model_noise)
        else:
            result = quote_from_request(request, patched_config, noise=sim.model_noise)
            baseline = quote_from_request(request, base_config, noise=sim.model_noise)

        carrier = carrier_api_feedback(request)
        label = _label_from_result(result)
        baseline_label = _label_from_result(baseline)
        carrier_label = _label_from_feedback(carrier)

        ok = label == carrier_label
        baseline_ok = baseline_label == carrier_label

        metrics.total += 1
        metrics.ok += 1 if ok else 0
        metrics.baseline_ok += 1 if baseline_ok else 0
        metrics.retrieved_patches += len(retrieved_patch_ids)
        metrics.applied_patches += len(applied_patch_ids)
        metrics.ok_window.append(1 if ok else 0)
        if len(metrics.ok_window) > 50:
            metrics.ok_window.pop(0)

        failure_cluster = None
        patch_log: List[str] = []
        if not ok:
            failure_cluster = carrier_label if carrier_label != "ok" else label
            metrics.record_failure(failure_cluster)
            patch_log.append(f"failure_label: {failure_cluster}")

            try:
                patch = _build_patch_from_feedback(request, result, carrier)
                patch.trigger_embedding = embedder.embed(patch.trigger).values
                metrics.patches_created += 1

                test_config = make_base_config()
                for existing in patch_store.patches:
                    if existing.status == "active":
                        _apply_patches(test_config, [existing])
                _apply_patches(test_config, [patch])

                tests = regression_suite + patch.tests
                if _run_tests(test_config, tests):
                    patch.status = "active"
                    patch_store.upsert(patch)
                    patch_log.append(
                        f"create_patch: {patch.patch_id} trigger='{patch.trigger}' status=active tests=pass"
                    )
                else:
                    patch.status = "quarantined"
                    patch_store.upsert(patch)
                    patch_log.append(
                        f"create_patch: {patch.patch_id} trigger='{patch.trigger}' status=quarantined tests=fail"
                    )
            except ValueError as exc:
                patch_log.append(f"create_patch: skipped ({exc})")

        trace = Trace(
            request=request,
            result=result,
            carrier_feedback=carrier,
            ok=ok,
            baseline_ok=baseline_ok,
            applied_patch_ids=applied_patch_ids,
            retrieved_patch_ids=retrieved_patch_ids,
            failure_cluster=failure_cluster,
        )
        trace_store.append(trace, query_vec)

        if sim.verbose and printed_traces < sim.trace_runs:
            _print_verbose_trace(
                run_idx=run_idx,
                request=request,
                baseline=baseline,
                baseline_steps=baseline_steps,
                result=result,
                patched_steps=patched_steps,
                carrier=carrier,
                retrieved_patch_ids=retrieved_patch_ids,
                applied_patch_ids=applied_patch_ids,
                patch_log=patch_log,
            )
            printed_traces += 1

    active, quarantined = patch_store.counts()
    metrics.patches_active = active
    metrics.patches_quarantined = quarantined
    golden_evals = _evaluate_golden_set(sim, embedder, patch_store)
    return metrics, [t for t, _ in trace_store.traces], patch_store, golden_evals


def _format_result(result: Result) -> str:
    if result.ok():
        quote = result.quote
        return f"OK zone={quote.zone} weight_kg={quote.weight_kg} cost={quote.cost}"
    return f"ERR {result.error.code} ({result.error.detail})"


def _print_verbose_trace(
    run_idx: int,
    request: str,
    baseline: Result,
    baseline_steps: List[str],
    result: Result,
    patched_steps: List[str],
    carrier: CarrierFeedback,
    retrieved_patch_ids: List[str],
    applied_patch_ids: List[str],
    patch_log: List[str],
) -> None:
    print("")
    print(f"=== TRACE RUN {run_idx} ===")
    print(f"request: {request}")
    print(f"retrieve patches: {retrieved_patch_ids if retrieved_patch_ids else '[]'}")
    print(f"apply patches: {applied_patch_ids if applied_patch_ids else '[]'}")
    if patch_log:
        print("patch events:")
        for entry in patch_log:
            print(f"  - {entry}")
    print("")
    print("[baseline]")
    for step in baseline_steps:
        print(f"  - {step}")
    print(f"  => {_format_result(baseline)}")
    print("")
    print("[patched]")
    for step in patched_steps:
        print(f"  - {step}")
    print(f"  => {_format_result(result)}")
    print("")
    carrier_label = _label_from_feedback(carrier)
    print(f"[carrier feedback] {carrier_label}")
