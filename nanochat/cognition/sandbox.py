"""Lightweight branch-and-score sandbox for candidate experimentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .creative import CreativeCandidate
from .normalize import overlap_score, term_set, unique_terms
from .verifier import RankedCandidate


@dataclass(slots=True)
class SandboxOutcome:
    candidate_id: str
    strategy_id: str
    branch: str
    score: float
    rationale: str
    verdict: str

    def as_trace_payload(self) -> dict[str, float | str]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "score": self.score,
            "rationale": self.rationale,
            "verdict": self.verdict,
        }


@dataclass(slots=True)
class SandboxReport:
    outcomes: list[SandboxOutcome] = field(default_factory=list)
    selected: SandboxOutcome | None = None


class LightweightSandbox:
    """Score shortlisted branches with transparent branch-vs-grounding heuristics."""

    def explore(
        self,
        query: str,
        candidates: list[CreativeCandidate],
        *,
        verifier_ranked: list[RankedCandidate] | None = None,
        support_profile: dict[str, object] | None = None,
    ) -> SandboxReport:
        query_terms = unique_terms(query)
        support_terms = set(str(term) for term in (support_profile or {}).get("support_terms", []))
        verifier_scores = {
            item.candidate_id: item.total_score
            for item in (verifier_ranked or [])
        }

        outcomes: list[SandboxOutcome] = []
        for candidate in candidates:
            candidate_terms = term_set(candidate.response)
            relevance = overlap_score(query_terms, candidate_terms)
            support_overlap = _overlap_from_set(support_terms, candidate_terms)
            verifier_score = verifier_scores.get(candidate.candidate_id, 0.0)
            branch_bonus = 0.15 if candidate.strategy_id in {"divergent_ideas", "branch_resolution"} else 0.0
            score = (0.45 * relevance) + (0.30 * verifier_score) + (0.15 * support_overlap) + branch_bonus
            verdict = "promote" if score >= 0.45 else "discard"
            outcomes.append(
                SandboxOutcome(
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_id,
                    branch=candidate.response,
                    score=round(score, 4),
                    rationale=(
                        f"relevance={relevance:.2f}; verifier={verifier_score:.2f}; "
                        f"support={support_overlap:.2f}; branch_bonus={branch_bonus:.2f}"
                    ),
                    verdict=verdict,
                )
            )

        outcomes.sort(key=lambda item: item.score, reverse=True)
        return SandboxReport(outcomes=outcomes, selected=outcomes[0] if outcomes else None)


def _overlap_from_set(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in candidate_terms)
    return hits / len(query_terms)
