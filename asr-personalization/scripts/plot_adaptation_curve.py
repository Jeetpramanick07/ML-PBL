"""Phase 4: plot the adaptation curve — WER vs. enrollment minutes, one line
per initialization strategy (naive / meta), with the Phase 3 generic model's
WER on that speaker as a flat reference line. One panel per speaker_holdout
TEST speaker, plus a mean-across-speakers summary panel.

Reads:
    reports/personalization_results.csv  (from run_personalization_experiment.py)
                                          (its "generic" rows, 0 enrollment, give the
                                           reference line on the same eval subset)
Writes:
    reports/adaptation_curve.png

Usage:
    python scripts/plot_adaptation_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = PROJECT_ROOT / "reports" / "personalization_results.csv"
OUT_PATH = PROJECT_ROOT / "reports" / "adaptation_curve.png"

STRATEGY_STYLE = {
    "naive": {"color": "tab:orange", "marker": "o", "label": "Naive per-speaker LoRA"},
    "meta": {"color": "tab:blue", "marker": "s", "label": "Meta-initialized per-speaker LoRA"},
}


def load_generic_wer_by_speaker(results: pd.DataFrame) -> dict[str, float]:
    """Generic-model (0 enrollment) WER per speaker, taken from the 'generic'
    rows the experiment script evaluates on the SAME eval subset as every
    adapter, so the reference line is directly comparable."""
    g = results[results.strategy == "generic"]
    return dict(zip(g.speaker_id, g.wer))


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found — run scripts/run_personalization_experiment.py first.")

    results = pd.read_csv(RESULTS_PATH)
    generic_wer = load_generic_wer_by_speaker(results)
    results = results[results.strategy != "generic"]
    speakers = sorted(results.speaker_id.unique())

    n_panels = len(speakers) + 1  # + one mean-across-speakers summary panel
    n_cols = min(3, n_panels)
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    def plot_panel(ax, sub: pd.DataFrame, title: str, ref_wer: float | None):
        for strategy, style in STRATEGY_STYLE.items():
            s = sub[sub.strategy == strategy].sort_values("enrollment_minutes")
            if s.empty:
                continue
            ax.plot(s.enrollment_minutes, s.wer, color=style["color"], marker=style["marker"], label=style["label"])
        if ref_wer is not None:
            ax.axhline(ref_wer, color="gray", linestyle="--", linewidth=1.5, label="Generic model (0 enrollment)")
        ax.set_title(title)
        ax.set_xlabel("Enrollment minutes")
        ax.set_ylabel("WER")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    for i, speaker_id in enumerate(speakers):
        sub = results[results.speaker_id == speaker_id]
        plot_panel(axes_flat[i], sub, f"Speaker {speaker_id}", generic_wer.get(speaker_id))

    mean_ref = sum(generic_wer.get(s) for s in speakers if s in generic_wer) / max(
        len([s for s in speakers if s in generic_wer]), 1
    ) if generic_wer else None
    mean_df = results.groupby(["strategy", "enrollment_minutes"], as_index=False)["wer"].mean()
    plot_panel(axes_flat[len(speakers)], mean_df, "Mean across held-out speakers", mean_ref)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Few-shot personalization: WER vs. enrollment minutes", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
