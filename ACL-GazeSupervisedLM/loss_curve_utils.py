import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Optional


LOSS_CURVE_COLUMNS = [
    "epoch",
    "split",
    "loss_total_mean",
    "loss_standard_mean",
    "loss_scanpath_mean",
    "augweight",
    "num_batches",
]


def _optional_float(value: Optional[float]):
    return math.nan if value is None else float(value)


def build_loss_curve_row(
    epoch: int,
    split: str,
    loss_total_mean: Optional[float],
    loss_standard_mean: Optional[float],
    loss_scanpath_mean: Optional[float],
    augweight: Optional[float],
    num_batches: int,
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "split": split,
        "loss_total_mean": _optional_float(loss_total_mean),
        "loss_standard_mean": _optional_float(loss_standard_mean),
        "loss_scanpath_mean": _optional_float(loss_scanpath_mean),
        "augweight": _optional_float(augweight),
        "num_batches": int(num_batches),
    }


def write_loss_curve_rows(output_dir: Path, rows: Iterable[Dict[str, object]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "loss_curves.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_CURVE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, math.nan) for column in LOSS_CURVE_COLUMNS})
    return path
