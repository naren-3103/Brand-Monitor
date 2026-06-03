import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.critic_models import (
    DimensionScores,
    CriticQAOutput,
    parse_scores_from_text,
    _parse_holistic_score,
)


@dataclass
class IterationRecord:
    iteration:        int
    synthesizer_output: str
    critic_output:    str          # user-facing markdown (feedback block included)
    quality_score:    float        # = dimension_scores.total when available
    dimension_scores: Optional[DimensionScores] = None
    improved:         bool = False  # True if score rose vs previous iteration


@dataclass
class FeedbackLoopResult:
    iterations:        List[IterationRecord] = field(default_factory=list)
    final_synthesizer: str = ""
    final_critic:      str = ""
    final_score:       float = 0.0
    converged:         bool = False

    def append(self, record: IterationRecord):
        record.improved = (
            len(self.iterations) == 0
            or record.quality_score > self.iterations[-1].quality_score
        )
        self.iterations.append(record)
        self.final_synthesizer = record.synthesizer_output
        self.final_critic      = record.critic_output
        self.final_score       = record.quality_score

    def score_progression(self) -> List[float]:
        return [r.quality_score for r in self.iterations]

    def dimension_progression(self) -> List[Optional[dict]]:
        """Per-iteration dimension breakdown dicts, for the Observability tab."""
        return [
            r.dimension_scores.as_display_dict() if r.dimension_scores else None
            for r in self.iterations
        ]


# ── Score extraction from a CrewAI TaskOutput object ─────────────────────────

def extract_scores_from_task_output(task_output) -> tuple:
    """
    Extract (quality_score, dimension_scores, critic_display_markdown) from a
    CrewAI TaskOutput.

    Priority:
      1. task_output.pydantic  → CriticQAOutput (structured, validated)
      2. task_output.raw text  → parse D1–D5 patterns (text fallback)
      3. task_output.raw text  → legacy holistic score parser (last resort)

    Returns:
      (float quality_score, Optional[DimensionScores], str critic_markdown)
    """
    if task_output is None:
        return 0.0, None, ""

    pydantic_obj: Optional[CriticQAOutput] = getattr(task_output, 'pydantic', None)

    if pydantic_obj is not None and isinstance(pydantic_obj, CriticQAOutput):
        score  = float(pydantic_obj.scores.total)
        dims   = pydantic_obj.scores
        md     = pydantic_obj.to_markdown()
        return score, dims, md

    # Fallback: text output
    raw_text: str = getattr(task_output, 'raw', '') or str(task_output)

    dims = parse_scores_from_text(raw_text)
    if dims.total > 0:
        score = float(dims.total)
    else:
        # Last resort: legacy holistic regex
        score = _parse_holistic_score(raw_text)
        dims  = None   # no structured dimension data

    return score, dims, raw_text


# ── Kept for backward compatibility (used by comparison_synthesizer) ──────────

def parse_quality_score(critic_text: str) -> float:
    """
    Extract a quality score from free-form critic text.
    Tries D1–D5 dimension sum first, then legacy holistic patterns.
    """
    dims = parse_scores_from_text(critic_text)
    if dims.total > 0:
        return float(dims.total)
    return _parse_holistic_score(critic_text)


def extract_critic_issues(critic_text: str) -> str:
    """
    Pull the 'Issues Found', 'Contradictions', and 'Feedback for Next Iteration'
    sections from the critic's markdown output so they can be injected into the
    next synthesizer prompt.  Falls back to the full text if sections not found.
    """
    sections = []
    for heading in ("Issues Found", "Contradictions", "Feedback for Next Iteration"):
        pattern = rf'(###\s*{re.escape(heading)}.*?)(?=###|\Z)'
        match   = re.search(pattern, critic_text, re.DOTALL | re.IGNORECASE)
        if match:
            sections.append(match.group(1).strip())

    return "\n\n".join(sections) if sections else critic_text.strip()
