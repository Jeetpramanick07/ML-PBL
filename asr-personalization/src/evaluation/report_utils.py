"""Shared evaluation/report helpers used by every eval script (Phase 2's
zero-shot baseline, Phase 3's generic fine-tune, and later phases): checkpointed
batched transcription over a manifest, WER/CER breakdown by group, per-speaker
tables, and markdown rendering. Kept in one place so every phase's eval report
uses identical breakdowns and is directly comparable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_wer_cer
from src.inference.whisper_infer import WhisperRunner


def load_cached_hyps(hyps_path: Path) -> dict[str, str]:
    if not hyps_path.exists():
        return {}
    cached = pd.read_csv(hyps_path)
    cached["hypothesis"] = cached["hypothesis"].fillna("")
    return dict(zip(cached["audio_path"], cached["hypothesis"]))


def flush_hyps(df: pd.DataFrame, results: dict[str, str], hyps_path: Path) -> None:
    done = df[df["audio_path"].isin(results.keys())].copy()
    done["hypothesis"] = done["audio_path"].map(results)
    hyps_path.parent.mkdir(parents=True, exist_ok=True)
    done.to_csv(hyps_path, index=False)


def evaluate_split(
    runner: WhisperRunner,
    df: pd.DataFrame,
    hyps_path: Path,
    batch_size: int,
    force: bool = False,
    checkpoint_every: int = 25,
    label: str = "",
) -> pd.DataFrame:
    """Transcribe every row of `df`, checkpointing progress to `hyps_path`
    every `checkpoint_every` utterances (keyed by audio_path) so a crash/
    OOM-kill only loses the last partial batch, and a re-run resumes."""
    results: dict[str, str] = {} if force else load_cached_hyps(hyps_path)

    all_paths = df["audio_path"].tolist()
    todo_paths = [p for p in all_paths if p not in results]

    if results:
        print(f"[{label}] resuming: {len(results)}/{len(df)} already transcribed, {len(todo_paths)} remaining")
    if not todo_paths:
        print(f"[{label}] all {len(df)} utterances already cached in {hyps_path}")
        out = df.copy()
        out["hypothesis"] = out["audio_path"].map(results)
        return out

    print(f"[{label}] transcribing {len(todo_paths)} utterances with {runner.model_name} on {runner.device} ...")
    t0 = time.time()
    since_checkpoint = 0
    for start in range(0, len(todo_paths), batch_size):
        batch_paths = todo_paths[start : start + batch_size]
        batch_hyps = runner.transcribe_batch(batch_paths)
        for p, h in zip(batch_paths, batch_hyps):
            results[p] = h
        since_checkpoint += len(batch_paths)
        done = start + len(batch_paths)
        if since_checkpoint >= checkpoint_every or done == len(todo_paths):
            flush_hyps(df, results, hyps_path)
            since_checkpoint = 0
            print(f"  checkpointed {len(results)}/{len(df)} total ({done}/{len(todo_paths)} this run)", flush=True)

    elapsed = time.time() - t0
    print(f"[{label}] done {len(todo_paths)} utterances in {elapsed:.1f}s "
          f"({elapsed / max(len(todo_paths), 1):.2f}s/utterance)")

    out = df.copy()
    out["hypothesis"] = out["audio_path"].map(results)
    return out


def group_metrics(df: pd.DataFrame, group_col: str | None = None) -> dict:
    if group_col is None:
        m = compute_wer_cer(df["transcription"].astype(str).tolist(), df["hypothesis"].astype(str).tolist())
        return {"n_samples": m.n_samples, "wer": m.wer, "cer": m.cer}

    result = {}
    for key, sub in df.groupby(group_col):
        m = compute_wer_cer(sub["transcription"].astype(str).tolist(), sub["hypothesis"].astype(str).tolist())
        result[str(key)] = {"n_samples": m.n_samples, "wer": m.wer, "cer": m.cer}
    return result


def summarize_split(df: pd.DataFrame) -> dict:
    summary = {"overall": group_metrics(df)}
    summary["by_dysarthria_status"] = group_metrics(df, "dysarthria_status")

    dysarthric = df[df.dysarthria_status == "dysarthric"]
    if dysarthric["severity"].notna().any():
        summary["by_severity"] = group_metrics(dysarthric.dropna(subset=["severity"]), "severity")

    return summary


def per_speaker_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for speaker_id, sub in df.groupby("speaker_id"):
        m = compute_wer_cer(sub["transcription"].astype(str).tolist(), sub["hypothesis"].astype(str).tolist())
        rows.append({
            "speaker_id": speaker_id,
            "dysarthria_status": sub["dysarthria_status"].iloc[0],
            "severity": sub["severity"].iloc[0] if pd.notna(sub["severity"].iloc[0]) else "",
            "n_samples": m.n_samples,
            "wer": m.wer,
            "cer": m.cer,
        })
    return pd.DataFrame(rows).sort_values("speaker_id").reset_index(drop=True)


def fmt_metrics_table(d: dict) -> str:
    lines = ["| group | n | WER | CER |", "|---|---|---|---|"]
    for key, m in d.items():
        lines.append(f"| {key} | {m['n_samples']} | {m['wer']:.4f} | {m['cer']:.4f} |")
    return "\n".join(lines)


def render_markdown(
    title: str,
    results: dict,
    per_speaker: dict[str, pd.DataFrame],
    warnings: list[str] | None = None,
) -> str:
    lines = [f"# {title}\n"]
    if warnings:
        lines.append("## WARNING: sanity-check issues\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    for split_name, summary in results.items():
        lines.append(f"## {split_name}\n")
        lines.append("### Overall\n")
        lines.append(fmt_metrics_table({"all": summary["overall"]}))
        lines.append("")
        lines.append("### By dysarthria status\n")
        lines.append(fmt_metrics_table(summary["by_dysarthria_status"]))
        lines.append("")
        if "by_severity" in summary:
            lines.append("### By severity (dysarthric speakers only)\n")
            lines.append(fmt_metrics_table(summary["by_severity"]))
            lines.append("")
        if split_name in per_speaker:
            lines.append("### Per held-out speaker\n")
            tbl = per_speaker[split_name]
            lines.append("| speaker_id | status | severity | n | WER | CER |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in tbl.iterrows():
                lines.append(
                    f"| {r.speaker_id} | {r.dysarthria_status} | {r.severity} | "
                    f"{r.n_samples} | {r.wer:.4f} | {r.cer:.4f} |"
                )
            lines.append("")

    return "\n".join(lines)
