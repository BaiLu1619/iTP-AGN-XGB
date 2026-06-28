from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_fasta(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seq_id: str | None = None
    label: int | None = None
    sequence_parts: list[str] = []

    def flush_record() -> None:
        if seq_id is None:
            return
        sequence = "".join(sequence_parts).upper()
        if not sequence:
            raise ValueError(f"empty sequence for {seq_id}")
        records.append({"id": seq_id, "label": int(label), "sequence": sequence})

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:]
                fields = header.split("|")
                if len(fields) < 2 or fields[1] not in {"0", "1"}:
                    raise ValueError(f"FASTA header must contain id|0 or id|1: {header}")
                seq_id = fields[0]
                label = int(fields[1])
                sequence_parts = []
            else:
                sequence_parts.append(line)
        flush_record()

    if not records:
        raise ValueError(f"no FASTA records found: {path}")
    return pd.DataFrame(records)


def save_predictions(
    path: str | Path,
    ids: list[str],
    y_true,
    y_pred,
    y_score,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ids,
            "true_label": y_true,
            "pred_label": y_pred,
            "probability": y_score,
        }
    ).to_csv(output_path, index=False)


def print_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

