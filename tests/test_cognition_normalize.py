from nanochat.cognition.normalize import normalize_terms, overlap_score, term_set, unique_terms


def test_normalize_terms_are_case_insensitive_and_ignore_punctuation_and_whitespace() -> None:
    first = normalize_terms("Please, summarize   THIS draft!!!")
    second = normalize_terms("please summarize this draft")

    assert first == ["summarization", "draft"]
    assert first == second


def test_alias_like_variants_share_overlapping_normalized_terms() -> None:
    first = unique_terms("Brainstorming ideas from earlier notes")
    second = term_set("Generate alternatives from previous notes")

    assert {"creative_explore", "memory_reuse", "notes"} <= set(first) & second
    assert overlap_score(first, second) == 1.0


def test_unique_terms_removes_duplicates_while_preserving_useful_tokens() -> None:
    terms = unique_terms("Brainstorm brainstorming ideas ideas from prior prior work")

    assert terms == ["creative_explore", "memory_reuse", "work"]


def test_term_set_collects_terms_from_strings_lists_and_dicts() -> None:
    terms = term_set(
        "Summarize this draft",
        ["Earlier summary", "Prior report"],
        {"style": "Terse bullet style summaries"},
    )

    assert {"summarization", "draft", "memory_reuse", "report", "style", "terse", "bullet"} <= terms
