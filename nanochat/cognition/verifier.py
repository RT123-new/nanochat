"""Verifier workspace for ranking and selecting creative candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

from .creative import CreativeCandidate
from .normalize import overlap_score, term_set, unique_terms


@dataclass(slots=True)
class RankedCandidate:
    candidate_id: str
    strategy_id: str
    candidate: str
    total_score: float
    relevance_score: float
    usefulness_score: float
    diversity_score: float
    repairability_score: float
    strategy_fit_score: float
    rationale: str
    repair_hint: str

    def as_trace_payload(self) -> dict[str, float | str]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "total_score": self.total_score,
            "relevance_score": self.relevance_score,
            "usefulness_score": self.usefulness_score,
            "diversity_score": self.diversity_score,
            "repairability_score": self.repairability_score,
            "strategy_fit_score": self.strategy_fit_score,
            "rationale": self.rationale,
            "repair_hint": self.repair_hint,
        }


@dataclass(slots=True)
class VerificationSelection:
    chosen: RankedCandidate
    ranked: list[RankedCandidate] = field(default_factory=list)
    repair_required: bool = False
    repair_reason: str = "not_needed"


class VerifierWorkspace:
    """Rank candidates with explicit relevance/diversity/usefulness heuristics."""

    def rank(
        self,
        query: str,
        candidates: list[CreativeCandidate],
        *,
        route: str = "creative_explore",
        support_profile: dict[str, object] | None = None,
    ) -> list[RankedCandidate]:
        query_terms = unique_terms(query)
        support_terms = set(str(term) for term in (support_profile or {}).get("support_terms", []))
        candidate_term_sets = {candidate.candidate_id: term_set(candidate.response) for candidate in candidates}

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            candidate_terms = candidate_term_sets.get(candidate.candidate_id, set())
            if not candidate.response.strip():
                ranked.append(
                    RankedCandidate(
                        candidate_id=candidate.candidate_id,
                        strategy_id=candidate.strategy_id,
                        candidate=candidate.response,
                        total_score=0.0,
                        relevance_score=0.0,
                        usefulness_score=0.0,
                        diversity_score=0.0,
                        repairability_score=0.0,
                        strategy_fit_score=0.0,
                        rationale="empty candidate",
                        repair_hint="regenerate with clearer grounding",
                    )
                )
                continue

            relevance = overlap_score(query_terms, candidate_terms)
            support_overlap = _overlap_from_set(support_terms, candidate_terms)
            usefulness = max(relevance, support_overlap)
            diversity = _diversity_score(candidate.candidate_id, candidate_term_sets)
            repairability = _repairability_score(candidate.response, relevance)
            strategy_fit = _strategy_fit_score(route, candidate.strategy_id, bool(support_terms))
            total = (
                0.35 * relevance
                + 0.20 * usefulness
                + 0.20 * diversity
                + 0.15 * repairability
                + 0.10 * strategy_fit
            )
            ranked.append(
                RankedCandidate(
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_id,
                    candidate=candidate.response,
                    total_score=round(total, 4),
                    relevance_score=round(relevance, 4),
                    usefulness_score=round(usefulness, 4),
                    diversity_score=round(diversity, 4),
                    repairability_score=round(repairability, 4),
                    strategy_fit_score=round(strategy_fit, 4),
                    rationale=(
                        f"relevance={relevance:.2f}; usefulness={usefulness:.2f}; "
                        f"diversity={diversity:.2f}; repairability={repairability:.2f}; "
                        f"strategy_fit={strategy_fit:.2f}"
                    ),
                    repair_hint=_repair_hint(relevance=relevance, usefulness=usefulness, candidate_text=candidate.response),
                )
            )
        ranked.sort(key=lambda item: (item.total_score, item.diversity_score, item.usefulness_score), reverse=True)
        return ranked

    def select(
        self,
        query: str,
        candidates: list[CreativeCandidate],
        *,
        route: str = "creative_explore",
        support_profile: dict[str, object] | None = None,
    ) -> VerificationSelection:
        ranked = self.rank(
            query=query,
            candidates=candidates,
            route=route,
            support_profile=support_profile,
        )
        if not ranked:
            empty = RankedCandidate(
                candidate_id="",
                strategy_id="",
                candidate="",
                total_score=0.0,
                relevance_score=0.0,
                usefulness_score=0.0,
                diversity_score=0.0,
                repairability_score=0.0,
                strategy_fit_score=0.0,
                rationale="no candidates",
                repair_hint="generate candidates before verification",
            )
            return VerificationSelection(chosen=empty, ranked=[], repair_required=True, repair_reason="no_candidates")

        chosen = ranked[0]
        repair_required = route == "verify" and (
            chosen.relevance_score < 0.35 or chosen.usefulness_score < 0.35
        )
        repair_reason = "insufficient_grounding" if repair_required else "not_needed"
        return VerificationSelection(
            chosen=chosen,
            ranked=ranked,
            repair_required=repair_required,
            repair_reason=repair_reason,
        )


def _overlap_from_set(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in candidate_terms)
    return hits / len(query_terms)


def _diversity_score(candidate_id: str, candidate_term_sets: dict[str, set[str]]) -> float:
    candidate_terms = candidate_term_sets.get(candidate_id, set())
    if not candidate_terms:
        return 0.0

    similarities = [
        _jaccard(candidate_terms, other_terms)
        for other_id, other_terms in candidate_term_sets.items()
        if other_id != candidate_id and other_terms
    ]
    if not similarities:
        return 0.5
    return max(0.0, 1.0 - max(similarities))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _repairability_score(candidate_text: str, relevance_score: float) -> float:
    token_count = len(candidate_text.split())
    if token_count == 0:
        return 0.0
    if token_count < 3:
        return 0.4
    return min(1.0, 0.5 + 0.5 * relevance_score)


def _strategy_fit_score(route: str, strategy_id: str, has_support_terms: bool) -> float:
    if route == "creative_explore" and strategy_id == "divergent_ideas":
        return 1.0
    if route == "verify" and strategy_id in {"branch_resolution", "recombination", "conservative_answer"}:
        return 1.0
    if route == "sandbox" and strategy_id in {"divergent_ideas", "branch_resolution"}:
        return 1.0
    if has_support_terms and strategy_id == "memory_grounded":
        return 1.0
    return 0.5


def _repair_hint(*, relevance: float, usefulness: float, candidate_text: str) -> str:
    if not candidate_text.strip():
        return "regenerate with concrete content"
    if relevance < 0.35:
        return "tighten the answer around the request"
    if usefulness < 0.35:
        return "ground the answer in retrieved support"
    return "keep the answer but trim redundancy if needed"
