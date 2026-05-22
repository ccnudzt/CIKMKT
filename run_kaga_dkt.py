import argparse
from pathlib import Path

from src.experiments.train import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KAGA-DKT experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a unified experiment config. Defaults to configs/experiments/kaga_dkt/<dataset>.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mooc_raw",
        choices=["mooc_raw", "xe_raw"],
        help="Dataset alias used by the default unified configs",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path("configs/experiments/kaga_dkt") / f"{args.dataset}.yaml"
    run_experiment(config_path)


if __name__ == "__main__":
    main()
