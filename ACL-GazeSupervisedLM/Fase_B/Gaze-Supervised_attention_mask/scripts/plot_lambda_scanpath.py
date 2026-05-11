import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot adaptive scanpath lambda from lambda_scanpath_log.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory that contains lambda_scanpath_log.csv.")
    parser.add_argument("--csv_name", default="lambda_scanpath_log.csv", help="CSV file name inside output_dir.")
    parser.add_argument("--plots_dir", default=None, help="Optional directory for PNG files. Defaults to output_dir/loss_plots.")
    return parser.parse_args()


def has_values(dataframe: pd.DataFrame, column: str) -> bool:
    return column in dataframe.columns and dataframe[column].notna().any()


def plot_lambda_by_epoch(dataframe: pd.DataFrame, target_path: Path) -> bool:
    if not has_values(dataframe, "lambda_scanpath_t"):
        return False

    plot_df = dataframe.sort_values("epoch")
    plt.figure(figsize=(8, 5))
    plt.plot(
        plot_df["epoch"],
        plot_df["lambda_scanpath_t"],
        marker="o",
        label="lambda usada en epoch",
    )
    if has_values(plot_df, "lambda_scanpath_next"):
        plt.plot(
            plot_df["epoch"],
            plot_df["lambda_scanpath_next"],
            marker="x",
            linestyle="--",
            label="lambda siguiente",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Lambda scanpath")
    plt.title("Lambda adaptativo scanpath por epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=160)
    plt.close()
    return True


def plot_progress_by_epoch(dataframe: pd.DataFrame, target_path: Path) -> bool:
    if not has_values(dataframe, "progress"):
        return False

    plot_df = dataframe.sort_values("epoch")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["epoch"], plot_df["progress"], marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Progress")
    plt.title("Progreso usado para actualizar lambda scanpath")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(target_path, dpi=160)
    plt.close()
    return True


def plot_scanpath_loss_and_lambda(dataframe: pd.DataFrame, target_path: Path) -> bool:
    if not has_values(dataframe, "lambda_scanpath_t"):
        return False
    if not has_values(dataframe, "current_scanpath_loss_for_update"):
        return False

    plot_df = dataframe.sort_values("epoch")
    fig, axis_lambda = plt.subplots(figsize=(8, 5))
    axis_loss = axis_lambda.twinx()

    axis_lambda.plot(
        plot_df["epoch"],
        plot_df["lambda_scanpath_t"],
        marker="o",
        color="tab:blue",
        label="lambda usada",
    )
    axis_loss.plot(
        plot_df["epoch"],
        plot_df["current_scanpath_loss_for_update"],
        marker="s",
        linestyle="--",
        color="tab:red",
        label="loss scanpath usada",
    )

    axis_lambda.set_xlabel("Epoch")
    axis_lambda.set_ylabel("Lambda scanpath", color="tab:blue")
    axis_loss.set_ylabel("Loss scanpath", color="tab:red")
    axis_lambda.tick_params(axis="y", labelcolor="tab:blue")
    axis_loss.tick_params(axis="y", labelcolor="tab:red")
    axis_lambda.grid(True, alpha=0.3)

    handles_lambda, labels_lambda = axis_lambda.get_legend_handles_labels()
    handles_loss, labels_loss = axis_loss.get_legend_handles_labels()
    axis_lambda.legend(handles_lambda + handles_loss, labels_lambda + labels_loss, loc="best")
    plt.title("Lambda adaptativo vs loss scanpath")
    fig.tight_layout()
    fig.savefig(target_path, dpi=160)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path = output_dir / args.csv_name
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "loss_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path)
    required = {"epoch", "lambda_scanpath_t"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    written_paths = []
    targets = [
        (plot_lambda_by_epoch, plots_dir / "lambda_scanpath_by_epoch.png"),
        (plot_progress_by_epoch, plots_dir / "lambda_scanpath_progress_by_epoch.png"),
        (plot_scanpath_loss_and_lambda, plots_dir / "lambda_scanpath_vs_loss_scanpath.png"),
    ]
    for plot_fn, target_path in targets:
        if plot_fn(dataframe, target_path):
            written_paths.append(target_path)

    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
