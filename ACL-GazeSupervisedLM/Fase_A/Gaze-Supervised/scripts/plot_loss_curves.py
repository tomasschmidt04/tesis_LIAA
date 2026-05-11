import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOSS_COLUMNS = {
    "loss_total_mean": "loss_total",
    "loss_standard_mean": "loss_standard",
    "loss_scanpath_mean": "loss_scanpath",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot epoch-level train/eval loss curves from loss_curves.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory that contains loss_curves.csv.")
    parser.add_argument("--csv_name", default="loss_curves.csv", help="CSV file name inside output_dir.")
    parser.add_argument("--plots_dir", default=None, help="Optional directory for PNG files. Defaults to output_dir/loss_plots.")
    return parser.parse_args()


def has_values(dataframe: pd.DataFrame, column: str) -> bool:
    return column in dataframe.columns and dataframe[column].notna().any()


def plot_train_eval(dataframe: pd.DataFrame, column: str, title: str, target_path: Path):
    plt.figure(figsize=(8, 5))
    for split in ["train", "eval"]:
        split_df = dataframe[dataframe["split"] == split].sort_values("epoch")
        if split_df.empty or not split_df[column].notna().any():
            continue
        plt.plot(split_df["epoch"], split_df[column], marker="o", label=split)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=150)
    plt.close()


def plot_all_losses(dataframe: pd.DataFrame, split: str, target_path: Path):
    split_df = dataframe[dataframe["split"] == split].sort_values("epoch")
    if split_df.empty:
        return False

    plotted_any = False
    plt.figure(figsize=(8, 5))
    for column, label in LOSS_COLUMNS.items():
        if has_values(split_df, column):
            plt.plot(split_df["epoch"], split_df[column], marker="o", label=label)
            plotted_any = True

    if not plotted_any:
        plt.close()
        return False

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"All losses - {split}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=150)
    plt.close()
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path = output_dir / args.csv_name
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "loss_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path)
    expected = {"epoch", "split", *LOSS_COLUMNS.keys()}
    missing = expected - set(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    written_paths = []
    for column, label in LOSS_COLUMNS.items():
        if has_values(dataframe, column):
            target_path = plots_dir / f"{label}_train_eval.png"
            plot_train_eval(dataframe, column, label, target_path)
            written_paths.append(target_path)

    for split in ["train", "eval"]:
        target_path = plots_dir / f"all_losses_{split}.png"
        if plot_all_losses(dataframe, split, target_path):
            written_paths.append(target_path)

    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
