from typing import Dict, List, Tuple

from .models import GoldenEval, Metrics, Patch, Trace


def _rate(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _precision_recall(evals: List[GoldenEval], pred_attr: str) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    labels = sorted({e.true_label for e in evals})
    stats: Dict[str, Tuple[float, float]] = {}
    precisions = []
    recalls = []

    for label in labels:
        tp = sum(1 for e in evals if e.true_label == label and getattr(e, pred_attr) == label)
        fp = sum(1 for e in evals if e.true_label != label and getattr(e, pred_attr) == label)
        fn = sum(1 for e in evals if e.true_label == label and getattr(e, pred_attr) != label)
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        stats[label] = (precision, recall)
        precisions.append(precision)
        recalls.append(recall)

    macro_p = _rate(sum(precisions), len(precisions))
    macro_r = _rate(sum(recalls), len(recalls))
    return stats, macro_p, macro_r


def _accuracy(evals: List[GoldenEval], pred_attr: str) -> float:
    return _rate(sum(1 for e in evals if getattr(e, pred_attr) == e.true_label), len(evals))


def _format_patch_details(patch: Patch) -> str:
    parts = []
    if patch.unit_conversions:
        parts.append("units=" + ",".join(sorted(patch.unit_conversions.keys())))
    if patch.dest_aliases:
        parts.append("dests=" + ",".join(sorted(patch.dest_aliases.keys())))
    if patch.item_aliases:
        parts.append("items=" + ",".join(sorted(patch.item_aliases.keys())))
    if patch.parcel_aliases:
        parts.append("parcels=" + ",".join(sorted(patch.parcel_aliases.keys())))
    if patch.prohibited_items:
        parts.append("prohibited=" + ",".join(sorted(patch.prohibited_items)))
    if patch.hazmat_items:
        parts.append("hazmat=" + ",".join(sorted(patch.hazmat_items)))
    if patch.liquid_items:
        parts.append("liquid=" + ",".join(sorted(patch.liquid_items)))
    if patch.embargo_dests:
        parts.append("embargo=" + ",".join(sorted(patch.embargo_dests)))
    if patch.parcel_max_kg:
        parts.append("parcel_max=" + ",".join(sorted(patch.parcel_max_kg.keys())))
    return ", ".join(parts) if parts else "-"


def format_report(
    metrics: Metrics,
    traces: List[Trace],
    patches: List[Patch],
    golden_evals: List[GoldenEval],
    show_patches: int = 5,
) -> str:
    patched_rate = _rate(metrics.ok, metrics.total)
    baseline_rate = _rate(metrics.baseline_ok, metrics.total)
    delta = patched_rate - baseline_rate
    rolling = _rate(sum(metrics.ok_window), len(metrics.ok_window))

    failure_lines = []
    for code, count in sorted(metrics.failures.items(), key=lambda x: x[1], reverse=True):
        failure_lines.append(f"- {code}: {count}")
    if not failure_lines:
        failure_lines = ["- none"]

    patch_lines = []
    for patch in patches[:show_patches]:
        details = _format_patch_details(patch)
        patch_lines.append(f"- {patch.patch_id} [{patch.status}] trigger='{patch.trigger}' ({details})")
    if not patch_lines:
        patch_lines = ["- none"]

    retrieved_per_run = _rate(metrics.retrieved_patches, metrics.total)
    applied_per_run = _rate(metrics.applied_patches, metrics.total)

    golden_lines = []
    if golden_evals:
        base_stats, base_macro_p, base_macro_r = _precision_recall(golden_evals, "baseline_label")
        patched_stats, patched_macro_p, patched_macro_r = _precision_recall(golden_evals, "patched_label")
        base_acc = _accuracy(golden_evals, "baseline_label")
        patched_acc = _accuracy(golden_evals, "patched_label")

        golden_lines.extend(
            [
                f"Golden set size: {len(golden_evals)}",
                f"Baseline golden accuracy: {base_acc:.2%}",
                f"Patched golden accuracy: {patched_acc:.2%}",
                f"Baseline macro P/R: {base_macro_p:.2%} / {base_macro_r:.2%}",
                f"Patched macro P/R: {patched_macro_p:.2%} / {patched_macro_r:.2%}",
                "Per-label precision/recall:",
            ]
        )
        for label in sorted(base_stats.keys()):
            b_p, b_r = base_stats[label]
            p_p, p_r = patched_stats[label]
            golden_lines.append(
                f"- {label}: baseline {b_p:.2%}/{b_r:.2%} | patched {p_p:.2%}/{p_r:.2%}"
            )
    else:
        golden_lines.append("Golden set size: 0")

    return "\n".join(
        [
            "=== PatchLab Simulation Report ===",
            f"Total runs: {metrics.total}",
            f"Baseline success rate: {baseline_rate:.2%}",
            f"Patched success rate: {patched_rate:.2%}",
            f"Delta: {delta:.2%}",
            f"Rolling(50) success rate: {rolling:.2%}",
            "",
            f"Patches created: {metrics.patches_created}",
            f"Active patches: {metrics.patches_active}",
            f"Quarantined patches: {metrics.patches_quarantined}",
            f"Avg patches retrieved/run: {retrieved_per_run:.2f}",
            f"Avg patches applied/run: {applied_per_run:.2f}",
            "",
            "Failure clusters:",
            *failure_lines,
            "",
            "Golden set evaluation:",
            *golden_lines,
            "",
            "Sample patches:",
            *patch_lines,
        ]
    )
