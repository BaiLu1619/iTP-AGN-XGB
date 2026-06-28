from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
DIPEPTIDES = [a + b for a, b in product(AA, repeat=2)]

AAINDEX = {
    "hydrophobicity": dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3])),
    "volume": dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6])),
    "polarity": dict(zip(AA, [8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9, 5.7, 11.6, 8.0, 10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2])),
    "charge": dict(zip(AA, [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])),
    "flexibility": dict(zip(AA, [0.36, 0.35, 0.51, 0.50, 0.31, 0.54, 0.32, 0.46, 0.47, 0.37, 0.30, 0.46, 0.51, 0.49, 0.53, 0.51, 0.44, 0.39, 0.31, 0.42])),
    "bulkiness": dict(zip(AA, [11.5, 13.5, 11.7, 13.6, 19.8, 3.4, 13.7, 21.4, 15.7, 21.4, 16.3, 12.8, 17.4, 14.5, 14.3, 9.5, 15.8, 21.6, 21.7, 18.0])),
    "isoelectric": dict(zip(AA, [6.0, 5.1, 2.8, 3.2, 5.5, 6.0, 7.6, 6.0, 9.7, 6.0, 5.7, 5.4, 6.3, 5.7, 10.8, 5.7, 5.6, 6.0, 5.9, 5.7])),
    "mass": dict(zip(AA, [89.1, 121.2, 133.1, 147.1, 165.2, 75.1, 155.2, 131.2, 146.2, 131.2, 149.2, 132.1, 115.1, 146.1, 174.2, 105.1, 119.1, 117.1, 204.2, 181.2])),
}

BLOSUM62 = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
    "F": [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3],
    "H": [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
    "I": [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
    "L": [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    "M": [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
    "N": [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
    "P": [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
    "Q": [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
    "R": [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
}

CTD_GROUPS = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normalized_vdw": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}


def clean_sequence(sequence: str) -> str:
    cleaned = "".join(aa for aa in sequence.upper() if aa in AA_SET)
    if not cleaned:
        raise ValueError(f"sequence contains no standard amino acids: {sequence}")
    return cleaned


def _aac(sequence: str) -> dict[str, float]:
    counts = Counter(sequence)
    length = len(sequence)
    return {f"AAC_{aa}": counts[aa] / length for aa in AA}


def _dpc(sequence: str, prefix: str = "DPC") -> dict[str, float]:
    pairs = [sequence[i : i + 2] for i in range(len(sequence) - 1)]
    counts = Counter(pairs)
    denom = max(len(pairs), 1)
    return {f"{prefix}_{dp}": counts[dp] / denom for dp in DIPEPTIDES}


def _asdc(sequence: str) -> dict[str, float]:
    counts = dict.fromkeys(DIPEPTIDES, 0.0)
    total = 0
    for i in range(len(sequence) - 1):
        for j in range(i + 1, len(sequence)):
            counts[sequence[i] + sequence[j]] += 1.0
            total += 1
    denom = max(total, 1)
    return {f"ASDC_{dp}": value / denom for dp, value in counts.items()}


def _aaindex(sequence: str) -> dict[str, float]:
    return {
        f"AAIndex_{name}": float(np.mean([values[aa] for aa in sequence]))
        for name, values in AAINDEX.items()
    }


def _sequence_order_correlation(sequence: str, lag: int, properties: list[dict[str, float]]) -> float:
    if len(sequence) <= lag:
        return 0.0
    values = []
    for i in range(len(sequence) - lag):
        diff_sum = sum((prop[sequence[i]] - prop[sequence[i + lag]]) ** 2 for prop in properties)
        values.append(diff_sum / len(properties))
    return float(np.mean(values))


def _paac(sequence: str, lambda_value: int = 5, weight: float = 0.05) -> dict[str, float]:
    lambda_value = min(lambda_value, max(len(sequence) - 1, 0))
    hydrophobicity = AAINDEX["hydrophobicity"]
    hydrophilicity = AAINDEX["polarity"]
    mass = AAINDEX["mass"]
    theta = [_sequence_order_correlation(sequence, lag, [hydrophobicity, hydrophilicity, mass]) for lag in range(1, lambda_value + 1)]
    counts = Counter(sequence)
    denom = 1.0 + weight * sum(theta)
    result = {f"PAAC_{aa}": (counts[aa] / len(sequence)) / denom for aa in AA}
    for i in range(5):
        value = theta[i] if i < len(theta) else 0.0
        result[f"PAAC_lambda_{i + 1}"] = (weight * value) / denom
    return result


def _apaac(sequence: str, lambda_value: int = 5, weight: float = 0.05) -> dict[str, float]:
    lambda_value = min(lambda_value, max(len(sequence) - 1, 0))
    hydrophobicity = AAINDEX["hydrophobicity"]
    hydrophilicity = AAINDEX["polarity"]
    theta_h = [_sequence_order_correlation(sequence, lag, [hydrophobicity]) for lag in range(1, lambda_value + 1)]
    theta_p = [_sequence_order_correlation(sequence, lag, [hydrophilicity]) for lag in range(1, lambda_value + 1)]
    theta = theta_h + theta_p
    counts = Counter(sequence)
    denom = 1.0 + weight * sum(theta)
    result = {f"APAAC_{aa}": (counts[aa] / len(sequence)) / denom for aa in AA}
    for i in range(5):
        result[f"APAAC_hydrophobicity_{i + 1}"] = (weight * (theta_h[i] if i < len(theta_h) else 0.0)) / denom
    for i in range(5):
        result[f"APAAC_hydrophilicity_{i + 1}"] = (weight * (theta_p[i] if i < len(theta_p) else 0.0)) / denom
    return result


def _ctd(sequence: str) -> dict[str, float]:
    result: dict[str, float] = {}
    length = len(sequence)
    for prop_name, groups in CTD_GROUPS.items():
        group_map = {aa: idx + 1 for idx, group in enumerate(groups) for aa in group}
        encoded = [group_map[aa] for aa in sequence]

        for group_id in (1, 2, 3):
            positions = [i + 1 for i, value in enumerate(encoded) if value == group_id]
            result[f"CTD_{prop_name}_C{group_id}"] = len(positions) / length
            for percentile in (0, 25, 50, 75, 100):
                if positions:
                    index = int(np.ceil(percentile / 100 * len(positions))) - 1
                    index = max(index, 0)
                    value = positions[index] / length
                else:
                    value = 0.0
                result[f"CTD_{prop_name}_D{group_id}_{percentile}"] = value

        transitions = {(1, 2): 0, (1, 3): 0, (2, 3): 0}
        for a, b in zip(encoded, encoded[1:]):
            if a == b:
                continue
            transitions[tuple(sorted((a, b)))] += 1
        denom = max(length - 1, 1)
        for pair, count in transitions.items():
            result[f"CTD_{prop_name}_T{pair[0]}{pair[1]}"] = count / denom
    return result


def _blosum62(sequence: str, max_len: int = 50) -> dict[str, float]:
    values: list[float] = []
    clipped = sequence[:max_len]
    for aa in clipped:
        values.extend(BLOSUM62[aa])
    values.extend([0.0] * ((max_len - len(clipped)) * len(AA)))
    return {f"BLOSUM62_{i + 1}": value for i, value in enumerate(values)}


FEATURE_BUILDERS = {
    "AAC": _aac,
    "DPC": _dpc,
    "ASDC": _asdc,
    "AAIndex": _aaindex,
    "PAAC": _paac,
    "APAAC": _apaac,
    "CTD": _ctd,
    "BLOSUM62": _blosum62,
}


def extract_features(records: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    if feature_name not in FEATURE_BUILDERS:
        raise ValueError(f"unsupported feature: {feature_name}")

    rows = []
    builder = FEATURE_BUILDERS[feature_name]
    for _, row in records.iterrows():
        sequence = clean_sequence(row["sequence"])
        features = builder(sequence)
        rows.append({"id": row["id"], "label": int(row["label"]), **features})
    return pd.DataFrame(rows)


def extract_all_features(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: extract_features(records, name) for name in FEATURE_BUILDERS}

