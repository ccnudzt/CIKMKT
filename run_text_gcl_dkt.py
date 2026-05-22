import argparse
from pathlib import Path

from src.experiments.train import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-GCL-DKT experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a unified experiment config. Defaults to configs/experiments/text_gcl_dkt/<dataset>_<variant>.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mooc_raw",
        choices=["mooc_raw", "xe_raw"],
        help="Dataset alias used by the default unified configs",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="random",
        choices=["random", "prob"],
        help="GCL augmentation strategy",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path("configs/experiments/text_gcl_dkt") / f"{args.dataset}_{args.variant}.yaml"
    run_experiment(config_path)


if __name__ == "__main__":
    main()
