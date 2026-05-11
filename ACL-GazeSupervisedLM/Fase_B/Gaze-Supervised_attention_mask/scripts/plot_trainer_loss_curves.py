import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot downstream Trainer loss curves from trainer_state.json.")
    parser.add_argument("--output_dir", required=True, help="Directory that contains trainer_state.json.")
    parser.add_argument("--state_name", default="trainer_state.json", help="Trainer state file name inside output_dir.")
    parser.add_argument("--csv_name", default="trainer_loss_curves.csv", help="CSV file written inside output_dir.")
    parser.add_argument("--plots_dir", default=None, help="Optional directory for PNG files. Defaults to output_dir/loss_plots.")
    return parser.parse_args()


def load_log_history(state_path: Path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("log_history", [])


def build_rows(log_history):
    rows = []
    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if "loss" in entry:
            rows.append(
                {
                    "split": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": entry["loss"],
                    "learning_rate": entry.get("learning_rate"),
                }
            )
        if "eval_loss" in entry:
            rows.append(
                {
                    "split": "eval",
                    "step": step,
                    "epoch": epoch,
                    "loss": entry["eval_loss"],
                    "learning_rate": None,
                }
            )
        if "test_loss" in entry:
            rows.append(
                {
                    "split": "test",
                    "step": step,
                    "epoch": epoch,
                    "loss": entry["test_loss"],
                    "learning_rate": None,
                }
            )
    return rows


def plot_loss_by_epoch(dataframe: pd.DataFrame, target_path: Path):
    plotted_any = False
    plt.figure(figsize=(8, 5))
    for split in ["train", "eval", "test"]:
        split_df = dataframe[dataframe["split"] == split].dropna(subset=["epoch", "loss"]).sort_values("epoch")
        if split_df.empty:
            continue
        plt.plot(split_df["epoch"], split_df["loss"], marker="o", label=split)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return False

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Downstream loss by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=150)
    plt.close()
    return True


def plot_train_loss_by_step(dataframe: pd.DataFrame, target_path: Path):
    train_df = dataframe[dataframe["split"] == "train"].dropna(subset=["step", "loss"]).sort_values("step")
    if train_df.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.plot(train_df["step"], train_df["loss"], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Downstream train loss by step")
    plt.tight_layout()
    plt.savefig(target_path, dpi=150)
    plt.close()
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    state_path = output_dir / args.state_name
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "loss_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(load_log_history(state_path))
    dataframe = pd.DataFrame(rows, columns=["split", "step", "epoch", "loss", "learning_rate"])
    csv_path = output_dir / args.csv_name
    dataframe.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    written_paths = []
    by_epoch_path = plots_dir / "loss_train_eval_test_by_epoch.png"
    if plot_loss_by_epoch(dataframe, by_epoch_path):
        written_paths.append(by_epoch_path)

    train_step_path = plots_dir / "loss_train_by_step.png"
    if plot_train_loss_by_step(dataframe, train_step_path):
        written_paths.append(train_step_path)

    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
