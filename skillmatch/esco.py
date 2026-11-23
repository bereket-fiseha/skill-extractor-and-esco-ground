"""Load the ESCO skills CSV into a flat, index-friendly structure."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ESCO_CSV


@dataclass
class EscoCatalog:
    """ESCO concepts plus a flat list of searchable labels pointing back to them."""

    concepts: pd.DataFrame          # one row per skill: uri, label, skill_type, description
    texts: list[str]                # searchable strings (preferred + alternative labels)
    text_to_concept: np.ndarray     # int32, same length as `texts`
    fingerprint: str                # changes whenever the CSV or label options change


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_catalog(csv_path: Path = ESCO_CSV, include_alt_labels: bool = True) -> EscoCatalog:
    df = pd.read_csv(
        csv_path,
        usecols=["conceptUri", "skillType", "preferredLabel", "altLabels", "description"],
    ).dropna(subset=["preferredLabel"]).reset_index(drop=True)
    df["description"] = df["description"].fillna("")
    df["altLabels"] = df["altLabels"].fillna("")

    texts: list[str] = []
    owners: list[int] = []
    for i, (pref, alts) in enumerate(zip(df["preferredLabel"], df["altLabels"])):
        pref = pref.strip()
        texts.append(pref)
        owners.append(i)
        if include_alt_labels and alts:
            seen = {pref.lower()}
            for alt in alts.split("\n"):
                alt = alt.strip()
                if alt and alt.lower() not in seen:
                    seen.add(alt.lower())
                    texts.append(alt)
                    owners.append(i)

    concepts = df.rename(
        columns={"conceptUri": "uri", "preferredLabel": "label", "skillType": "skill_type"}
    )[["uri", "label", "skill_type", "description"]]
    fingerprint = f"{_file_sha1(csv_path)}-{'alt' if include_alt_labels else 'pref'}"
    return EscoCatalog(concepts, texts, np.asarray(owners, dtype=np.int32), fingerprint)
