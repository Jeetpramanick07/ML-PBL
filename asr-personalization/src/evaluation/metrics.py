"""WER/CER computation for ASR evaluation.

Both reference and hypothesis text are run through the exact same
normalization used for training targets in
`src.data.preprocessing.normalize_transcription` (lowercase, strip
punctuation, collapse whitespace) before scoring. This is the only
normalization applied — no number/abbreviation expansion, no stopword
removal. Keeping this in sync with `preprocessing.py` matters: any drift
between the two silently inflates or deflates reported WER on differences
that aren't real recognition errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from src.data.preprocessing import normalize_transcription


@dataclass
class WerCer:
    wer: float
    cer: float
    n_samples: int


def _clean_pair(references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
    refs = [normalize_transcription(r) for r in references]
    hyps = [normalize_transcription(h) for h in hypotheses]
    # jiwer chokes on empty strings on both sides of a pair; keep the pair but
    # substitute a placeholder token so it counts as a full miss rather than
    # crashing the whole batch.
    refs = [r if r else "<empty>" for r in refs]
    hyps = [h if h else "<empty>" for h in hyps]
    return refs, hyps


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> WerCer:
    """Compute corpus-level WER and CER over a list of (reference, hypothesis) pairs."""
    if len(references) != len(hypotheses):
        raise ValueError(f"references ({len(references)}) and hypotheses ({len(hypotheses)}) length mismatch")
    if not references:
        return WerCer(wer=float("nan"), cer=float("nan"), n_samples=0)

    refs, hyps = _clean_pair(references, hypotheses)
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return WerCer(wer=round(wer, 4), cer=round(cer, 4), n_samples=len(references))
