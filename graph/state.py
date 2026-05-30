from typing import TypedDict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScoredSource:
    """A search result with quality metadata attached."""
    content: str
    url: str = ""
    domain: str = ""
    relevance_score: float = 0.0
    quality_score: float = 0.0
    composite_score: float = 0.0
    kept: bool = True
    reason: str = ""


@dataclass
class Contradiction:
    topic: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str


@dataclass
class CriticReport:
    kept: List[ScoredSource] = field(default_factory=list)
    dropped: List[ScoredSource] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    total_input: int = 0
    total_kept: int = 0
    total_dropped: int = 0


class ResearchState(TypedDict, total=False):
    # ── existing fields (unchanged) ──────────────────────────────
    query: str
    sub_questions: List[str]
    search_results: List[str]
    validated_results: List[str]
    report: str
    reflection: str
    search_queries: List[str]
    iteration: int

    # ── new fields added by upgraded agents ──────────────────────
    dropped_results: List[str]          # what the critic removed + why
    contradictions: List[dict]          # surfaced conflicts between sources
    citations: List[str]                # URLs actually cited in the report
    critic_report: Optional[CriticReport]  # full audit object for UI display