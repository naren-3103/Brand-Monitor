"""
Pydantic models for the Critic QA Agent's structured output.

Using output_pydantic on the critic task ensures:
- Scores are integers (0-2 per dimension, max 10 total) -- not free-form LLM floats
- All output fields are validated before the feedback loop sees them
- Dimension breakdown is machine-readable for the Observability tab
- No regex brittle-ness: score is accessed as .scores.total, not parsed from prose
"""
import re
from pydantic import BaseModel, Field
from typing import List, Optional


DIMENSION_LABELS = {
    'factual_accuracy':       'D1 Factual Accuracy',
    'claim_support':          'D2 Claim Support',
    'contradiction_handling': 'D3 Contradiction Handling',
    'recommendation_quality': 'D4 Recommendation Quality',
    'executive_completeness': 'D5 Executive Completeness',
}


class DimensionScores(BaseModel):
    """Five 0-2 dimension scores that sum to the quality score out of 10."""
    factual_accuracy:       int = Field(default=0, ge=0, le=2)
    claim_support:          int = Field(default=0, ge=0, le=2)
    contradiction_handling: int = Field(default=0, ge=0, le=2)
    recommendation_quality: int = Field(default=0, ge=0, le=2)
    executive_completeness: int = Field(default=0, ge=0, le=2)

    @property
    def total(self) -> int:
        return (
            self.factual_accuracy
            + self.claim_support
            + self.contradiction_handling
            + self.recommendation_quality
            + self.executive_completeness
        )

    def as_display_dict(self) -> dict:
        """Return {label: score} ordered dict for the Observability chart."""
        return {DIMENSION_LABELS[k]: getattr(self, k) for k in DIMENSION_LABELS}


class CriticQAOutput(BaseModel):
    """Fully structured output from the Critic QA Agent."""
    scores:                      DimensionScores
    strengths:                   List[str] = Field(default_factory=list)
    issues:                      List[str] = Field(default_factory=list)
    contradictions:              List[str] = Field(default_factory=list)
    executive_summary:           str = Field(default="")
    feedback_for_next_iteration: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """
        Convert to user-facing Markdown.

        Section headings are intentionally kept compatible with
        extract_critic_issues() so feedback injection still works.
        """
        lines = [
            "## Critic QA Agent",
            "",
            "### Quality Score",
            f"**{self.scores.total}/10**",
            # "",
            # "| Dimension | Score | Max |",
            # "|-----------|:-----:|:---:|",
        ]
        # for label, score in self.scores.as_display_dict().items():
        #     bar = "█" * score + "" * (2 - score)
        #     lines.append(f"| {label} | {score} `{bar}` | 2 |")
        # lines.append(f"| **Total** | **{self.scores.total}** | **10** |")
        # lines.append("")

        if self.strengths:
            lines += ["### Strengths", ""]
            for s in self.strengths:
                lines.append(f"- {s}")
            lines.append("")

        lines += ["### Issues Found", ""]
        if self.issues:
            for issue in self.issues:
                lines.append(f"- {issue}")
        else:
            lines.append("No issues found.")
        lines.append("")

        lines += ["### Contradictions", ""]
        if self.contradictions:
            for c in self.contradictions:
                lines.append(f"- {c}")
        else:
            lines.append("No contradictions detected.")
        lines.append("")

        if self.executive_summary:
            lines += ["### Executive QA Summary", "", self.executive_summary, ""]

        if self.feedback_for_next_iteration:
            lines += ["### Feedback for Next Iteration", ""]
            for fb in self.feedback_for_next_iteration:
                lines.append(f"- {fb}")
            lines.append("")

        return "\n".join(lines)


# ── Text fallback (when output_pydantic parsing fails) ───────────────────────

def parse_scores_from_text(text: str) -> DimensionScores:
    """
    Parse D1–D5 dimension scores from structured text.

    Looks for patterns like:
      D1_FACTUAL_ACCURACY: 2   |   D1: 2   |   D1 Factual Accuracy: 2

    Falls back to distributing a holistic score evenly across dimensions
    if no D1–D5 markers are found.
    """
    field_patterns = {
        'factual_accuracy':       r'D1[\w\s]*:\s*([0-2])',
        'claim_support':          r'D2[\w\s]*:\s*([0-2])',
        'contradiction_handling': r'D3[\w\s]*:\s*([0-2])',
        'recommendation_quality': r'D4[\w\s]*:\s*([0-2])',
        'executive_completeness': r'D5[\w\s]*:\s*([0-2])',
    }
    scores: dict = {}
    any_found = False
    for field, pattern in field_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            scores[field] = min(2, max(0, int(m.group(1))))
            any_found = True
        else:
            scores[field] = 0

    if not any_found:
        holistic = _parse_holistic_score(text)
        per_dim = max(0, min(2, round(holistic / 5)))
        return DimensionScores(**{k: per_dim for k in field_patterns})

    return DimensionScores(**scores)


def _parse_holistic_score(text: str) -> float:
    """Legacy: extract a single out-of-10 number from free-form critic text."""
    patterns = [
        r'###\s*Quality\s*Score\s*[\r\n]+\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',
        r'score[:\s]+(\d+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 10:
                return v
    return 0.0
