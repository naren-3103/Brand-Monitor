import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IterationRecord:
    iteration: int
    synthesizer_output: str
    critic_output: str
    quality_score: float
    improved: bool = False  # True if score rose vs previous iteration


@dataclass
class FeedbackLoopResult:
    iterations: List[IterationRecord] = field(default_factory=list)
    final_synthesizer: str = ""
    final_critic: str = ""
    final_score: float = 0.0
    converged: bool = False  # True if quality_threshold was reached

    def append(self, record: IterationRecord):
        record.improved = (
            len(self.iterations) == 0 or record.quality_score > self.iterations[-1].quality_score
        )
        self.iterations.append(record)
        self.final_synthesizer = record.synthesizer_output
        self.final_critic = record.critic_output
        self.final_score = record.quality_score

    def score_progression(self) -> List[float]:
        return [r.quality_score for r in self.iterations]


def parse_quality_score(critic_text: str) -> float:
    """
    Extract the numeric quality score (out of 10) from the critic's output.

    Handles formats:
      ### Quality Score
      8/10  |  8.5 / 10  |  8 out of 10  |  Score: 8
    Returns 0.0 if no score can be parsed.
    """
    # Prefer the number that immediately follows the Quality Score heading
    heading_match = re.search(
        r'###\s*Quality\s*Score\s*[\r\n]+\s*(\d+(?:\.\d+)?)',
        critic_text,
        re.IGNORECASE,
    )
    if heading_match:
        score = float(heading_match.group(1))
        if 0 <= score <= 10:
            return score

    # Fall back: any "N/10" or "N out of 10" pattern anywhere in the text
    fallback_patterns = [
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',
        r'score[:\s]+(\d+(?:\.\d+)?)',
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, critic_text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 10:
                return score

    return 0.0


def extract_critic_issues(critic_text: str) -> str:
    """
    Pull the '### Issues Found' and '### Contradictions' sections from the
    critic's output so they can be injected into the next synthesizer prompt.
    Falls back to the full critic text if those sections are not found.
    """
    sections = []
    for heading in ("Issues Found", "Contradictions", "Feedback for Next Iteration"):
        pattern = rf'(###\s*{re.escape(heading)}.*?)(?=###|\Z)'
        match = re.search(pattern, critic_text, re.DOTALL | re.IGNORECASE)
        if match:
            sections.append(match.group(1).strip())

    if sections:
        return "\n\n".join(sections)

    # If structured sections weren't found, return the full text
    return critic_text.strip()
