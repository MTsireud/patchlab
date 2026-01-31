import argparse

from patchlab.engine import SimulationConfig, run_simulation
from patchlab.measurement import format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchLab: retrieval + patches simulation")
    parser.add_argument("--runs", type=int, default=500, help="number of simulated requests")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--show", type=int, default=5, help="number of sample patches to show")
    parser.add_argument("--verbose", action="store_true", help="print extra details")
    parser.add_argument("--trace", type=int, default=5, help="number of verbose traces to print")
    parser.add_argument("--noise", type=float, default=0.03, help="model noise rate")
    parser.add_argument("--golden", type=int, default=40, help="golden set size for P/R eval")
    args = parser.parse_args()

    sim = SimulationConfig(
        runs=args.runs,
        seed=args.seed,
        show_patches=args.show,
        verbose=args.verbose,
        trace_runs=args.trace,
        model_noise=args.noise,
        golden_size=args.golden,
    )
    metrics, traces, patch_store, golden_evals = run_simulation(sim)
    print(format_report(metrics, traces, patch_store.patches, golden_evals, show_patches=args.show))


if __name__ == "__main__":
    main()
