"""Post-hoc train/test audio-overlap breakdown for the generic_full model.

pooled_* splits are a random utterance-level split over ALL speakers, while
speaker_holdout_test holds every utterance of F03/FC01/FC02/M04/M05. So the
exact audio files in pooled_train_subsample.csv (trained on) and
pooled_val.csv (used for best-checkpoint selection) partly ARE
speaker_holdout_test rows. This script does not change any eval protocol: it
re-scores already-saved hypotheses, split by whether each test row's audio
was trained on / validated on / never seen, so the headline numbers can be
read honestly.

Usage:
    python scripts/leakage_breakdown.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.preprocessing import normalize_transcription
from src.evaluation.metrics import compute_wer_cer

SPLITS = Path("data/processed/splits")
REPORTS = Path("reports")


def overlap_label(paths: pd.Series, train: set, val: set) -> pd.Series:
    return paths.map(lambda p: "trained_on" if p in train else ("val_only" if p in val else "clean"))


def score(df: pd.DataFrame, ref_col: str) -> dict:
    if len(df) == 0:
        return {"n": 0, "wer": None, "cer": None}
    m = compute_wer_cer(df[ref_col].astype(str).tolist(), df["hypothesis"].fillna("").astype(str).tolist())
    return {"n": int(m.n_samples), "wer": m.wer, "cer": m.cer}


def breakdown(df: pd.DataFrame, ref_col: str, keys: list[str]) -> list[dict]:
    rows = []
    for key in keys:
        groups = [("all", df)] if key == "all" else list(df.groupby(key))
        for gval, gdf in groups:
            for subset in ("all", "clean", "val_only", "trained_on"):
                sdf = gdf if subset == "all" else gdf[gdf.overlap == subset]
                rows.append({"group_by": key, "group": gval, "subset": subset, **score(sdf, ref_col)})
    return rows


def main() -> None:
    train = set(pd.read_csv(SPLITS / "pooled_train_subsample.csv").audio_path)
    val = set(pd.read_csv(SPLITS / "pooled_val.csv").audio_path)
    out: dict = {"holdout_test": {}, "personalization": []}

    ho = pd.read_csv(SPLITS / "speaker_holdout_test.csv")
    ho_labels = overlap_label(ho.audio_path, train, val)
    out["holdout_test_overlap_counts"] = ho_labels.value_counts().to_dict()
    out["holdout_test_overlap_by_speaker"] = pd.crosstab(ho.speaker_id, ho_labels).to_dict(orient="index")

    for arm, fname in (("baseline", "baseline_hyps_speaker_holdout_test.csv"),
                       ("generic_full", "generic_full_hyps_speaker_holdout_test.csv")):
        path = REPORTS / fname
        if not path.exists():
            print(f"skip {arm}: {path} missing")
            continue
        h = pd.read_csv(path)
        assert len(h) == len(ho), f"{fname} has {len(h)} rows, expected {len(ho)} (incomplete eval?)"
        h["overlap"] = overlap_label(h.audio_path, train, val)
        out["holdout_test"][arm] = breakdown(h, "transcription", ["all", "dysarthria_status", "speaker_id"])

    # Personalization eval_holdout sets: hyps files hold (reference, hypothesis)
    # in eval_holdout.csv row order; only valid if no rows were dropped.
    hyps_dir = REPORTS / "personalization_hyps_final"
    for hyp_path in sorted(hyps_dir.glob("*.csv")) if hyps_dir.exists() else []:
        speaker, strategy, minutes = hyp_path.stem.split("_", 2)
        ev = pd.read_csv(Path("data/processed/enrollment") / speaker / "eval_holdout.csv")
        h = pd.read_csv(hyp_path)
        aligned = len(h) == len(ev) and all(
            normalize_transcription(str(a)) == normalize_transcription(str(b))
            for a, b in zip(h.reference, ev.transcription)
        )
        if not aligned:
            print(f"WARNING: {hyp_path.name} not row-aligned with eval_holdout.csv; skipping")
            continue
        h["overlap"] = overlap_label(ev.audio_path, train, val).values
        for subset in ("all", "clean", "val_only", "trained_on"):
            sdf = h if subset == "all" else h[h.overlap == subset]
            out["personalization"].append({"speaker_id": speaker, "strategy": strategy, "minutes": minutes,
                                           "subset": subset, **score(sdf, "reference")})

    (REPORTS / "leakage_breakdown.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = ["# Train/test audio-overlap breakdown (generic_full)\n",
             "`clean` = audio never used in training or validation; `val_only` = in pooled_val "
             "(used for best-checkpoint selection, not gradient updates); `trained_on` = in "
             "pooled_train_subsample.csv.\n",
             f"speaker_holdout_test overlap counts: {out['holdout_test_overlap_counts']}\n"]
    for arm, rows in out["holdout_test"].items():
        lines += [f"## speaker_holdout_test — {arm}\n", "| group_by | group | subset | n | WER | CER |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['group_by']} | {r['group']} | {r['subset']} | {r['n']} | "
                  f"{r['wer']:.4f} | {r['cer']:.4f} |" for r in rows if r["n"]]
        lines.append("")
    if out["personalization"]:
        lines += ["## Personalization eval_holdout sets\n", "| speaker | strategy | min | subset | n | WER | CER |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['speaker_id']} | {r['strategy']} | {r['minutes']} | {r['subset']} | {r['n']} | "
                  f"{r['wer']:.4f} | {r['cer']:.4f} |" for r in out["personalization"] if r["n"]]
    (REPORTS / "leakage_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
