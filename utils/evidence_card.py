from dataclasses import dataclass
from typing import List

@dataclass
class EvidenceCard:
    agent_name: str
    insight: str
    evidence: List[str]
    confidence: str
    recommendation: str

    def to_markdown(self):
        evidence_text = "\n".join([f"- {e}" for e in self.evidence])

        return f"""
## {self.agent_name}

### Insight
{self.insight}

### Evidence
{evidence_text}

### Confidence
{self.confidence}

### Recommendation
{self.recommendation}
"""