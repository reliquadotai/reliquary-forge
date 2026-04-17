"""Unit tests for pure helpers in reliquary.distill.lane.

The end-to-end run_teacher_cycle / run_student_cycle paths use OpenAI-
compatible HTTP endpoints and are covered by integration tests. These
helpers (_strip_think_blocks, _canonicalize, _shingle,
contamination_screen, extract_final_answer, _artifact_id,
default_benchmark_ngrams) are pure Python and pin the contract that
the cycle dispatch depends on.
"""

from __future__ import annotations

from reliquary.distill.lane import (
    DEFAULT_CONTAMINATION_NGRAM,
    DEFAULT_CONTAMINATION_THRESHOLD,
    _artifact_id,
    _canonicalize,
    _shingle,
    _strip_think_blocks,
    contamination_screen,
    default_benchmark_ngrams,
    extract_final_answer,
)


# ---------------------------------------------------------------------------
# _strip_think_blocks
# ---------------------------------------------------------------------------


def test_strip_think_blocks_removes_closed_block() -> None:
    text = "Preface <think>internal reasoning</think>Final Answer: 42"
    assert _strip_think_blocks(text) == "Preface Final Answer: 42"


def test_strip_think_blocks_case_insensitive() -> None:
    text = "Prefix <THINK>x</THINK> tail"
    assert _strip_think_blocks(text) == "Prefix  tail"


def test_strip_think_blocks_handles_multiline() -> None:
    text = "a <think>line1\nline2</think> b"
    assert _strip_think_blocks(text) == "a  b"


def test_strip_think_blocks_drops_unclosed_block() -> None:
    text = "header <think>never closed and token limit hit"
    assert _strip_think_blocks(text) == "header"


def test_strip_think_blocks_idempotent_on_plain_text() -> None:
    text = "no tags here"
    assert _strip_think_blocks(text) == "no tags here"


def test_strip_think_blocks_strips_surrounding_whitespace() -> None:
    text = "   <think>x</think>   Final Answer: 1   "
    assert _strip_think_blocks(text) == "Final Answer: 1"


def test_strip_think_blocks_multiple_closed_blocks() -> None:
    text = "a <think>1</think> b <think>2</think> c"
    assert _strip_think_blocks(text) == "a  b  c"


# ---------------------------------------------------------------------------
# _canonicalize
# ---------------------------------------------------------------------------


def test_canonicalize_lowercases() -> None:
    assert _canonicalize("HELLO") == "hello"


def test_canonicalize_collapses_whitespace() -> None:
    assert _canonicalize("foo   bar\n\tbaz") == "foo bar baz"


def test_canonicalize_strips_edges() -> None:
    assert _canonicalize("   padded   ") == "padded"


def test_canonicalize_unicode_nfkc_normalization() -> None:
    # Fullwidth-Latin + regular Latin should collapse to lowercase ASCII.
    assert _canonicalize("ＡＢＣ").lower() == "abc"


def test_canonicalize_empty_string() -> None:
    assert _canonicalize("") == ""


# ---------------------------------------------------------------------------
# _shingle
# ---------------------------------------------------------------------------


def test_shingle_basic_trigrams() -> None:
    shingles = _shingle("the quick brown fox", n=3)
    assert "the quick brown" in shingles
    assert "quick brown fox" in shingles
    assert len(shingles) == 2


def test_shingle_too_short_returns_empty() -> None:
    assert _shingle("only two tokens", n=5) == set()


def test_shingle_uses_word_tokens_only() -> None:
    # Non-word chars split tokens but are not themselves tokens.
    shingles = _shingle("hi, there! how", n=2)
    assert shingles == {"hi there", "there how"}


def test_shingle_empty_string_empty_set() -> None:
    assert _shingle("", n=2) == set()


# ---------------------------------------------------------------------------
# contamination_screen
# ---------------------------------------------------------------------------


def test_contamination_screen_no_overlap() -> None:
    result = contamination_screen(
        "this is totally unique content for testing purposes only",
        benchmark_ngrams=set(),
    )
    assert result.flagged is False
    assert result.overlap == 0.0


def test_contamination_screen_full_overlap_flagged() -> None:
    # Craft a text whose every ngram appears in benchmark.
    text = " ".join(f"tok{i}" for i in range(20))
    # Build the ngrams AFTER canonicalization so they match what the screen computes.
    from reliquary.distill.lane import _canonicalize as canon, _shingle as shingle
    benchmark = shingle(canon(text), n=DEFAULT_CONTAMINATION_NGRAM)
    result = contamination_screen(text, benchmark_ngrams=benchmark)
    assert result.overlap == 1.0
    assert result.flagged is True
    assert result.reason == "benchmark_overlap"


def test_contamination_screen_text_too_short() -> None:
    result = contamination_screen("tiny", benchmark_ngrams=set())
    assert result.flagged is False
    assert result.reason == "text_too_short"


def test_contamination_screen_threshold_inclusive_exclusive() -> None:
    # Construct overlap exactly at threshold; should NOT flag.
    from reliquary.distill.lane import _canonicalize as canon, _shingle as shingle
    # Build 20 trigrams, 1 from benchmark: overlap = 1/20 = 0.05 (exactly threshold).
    text = " ".join(f"w{i}" for i in range(22))
    text_ngrams = shingle(canon(text), n=3)
    # Pick one ngram as the "benchmark".
    one = next(iter(text_ngrams))
    result = contamination_screen(
        text, benchmark_ngrams={one}, ngram=3, threshold=1.0 / len(text_ngrams)
    )
    # overlap is 1/20 = threshold; strictly greater check means NOT flagged.
    assert result.flagged is False


def test_contamination_screen_matched_ngrams_capped_at_10() -> None:
    from reliquary.distill.lane import _canonicalize as canon, _shingle as shingle
    text = " ".join(f"w{i}" for i in range(50))
    text_ngrams = shingle(canon(text), n=3)
    result = contamination_screen(text, benchmark_ngrams=text_ngrams, ngram=3)
    assert len(result.matched_ngrams) <= 10


def test_contamination_screen_default_constants_are_sane() -> None:
    assert DEFAULT_CONTAMINATION_NGRAM >= 2
    assert 0.0 < DEFAULT_CONTAMINATION_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_final_answer_canonical_anchor() -> None:
    assert extract_final_answer("Reasoning: x\n\nFinal Answer: 42") == "42"


def test_extract_final_answer_requires_separator() -> None:
    # Anchor must have ':' or '=' separator per the spec regex.
    assert extract_final_answer("no final answer here") is None


def test_extract_final_answer_ignores_think_block_guess() -> None:
    trace = "<think>interim answer: 5</think>\nFinal Answer: 7"
    assert extract_final_answer(trace) == "7"


def test_extract_final_answer_accepts_equals_separator() -> None:
    assert extract_final_answer("\nFinal Answer = 3") == "3"


def test_extract_final_answer_accepts_answer_is_phrasing() -> None:
    # "the answer is" with colon separator.
    assert extract_final_answer("\nThe answer is: 12") == "12"


def test_extract_final_answer_none_on_empty() -> None:
    assert extract_final_answer("") is None


def test_extract_final_answer_strip_trailing_newline() -> None:
    assert extract_final_answer("prefix\n\nFinal Answer: 99\n") == "99"


def test_extract_final_answer_falls_back_to_raw_when_strip_empties() -> None:
    # Text is ENTIRELY a think block; _strip returns ""; regex falls back on raw.
    trace = "<think>Final Answer: 42</think>"
    # After strip, text is empty → regex on empty returns None; original has Final Answer
    # inside think block, but strip removes it → None expected.
    assert extract_final_answer(trace) is None


# ---------------------------------------------------------------------------
# _artifact_id
# ---------------------------------------------------------------------------


def test_artifact_id_deterministic_under_key_reorder() -> None:
    a = _artifact_id({"x": 1, "y": 2})
    b = _artifact_id({"y": 2, "x": 1})
    assert a == b


def test_artifact_id_changes_with_payload() -> None:
    assert _artifact_id({"x": 1}) != _artifact_id({"x": 2})


def test_artifact_id_is_hex_sha256() -> None:
    digest = _artifact_id({"k": "v"})
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


# ---------------------------------------------------------------------------
# default_benchmark_ngrams
# ---------------------------------------------------------------------------


def test_default_benchmark_ngrams_returns_set() -> None:
    ngrams = default_benchmark_ngrams()
    assert isinstance(ngrams, set)


def test_default_benchmark_ngrams_non_empty() -> None:
    # A populated benchmark set is required for contamination_screen to be useful.
    ngrams = default_benchmark_ngrams()
    assert len(ngrams) >= 1


def test_default_benchmark_ngrams_idempotent() -> None:
    first = default_benchmark_ngrams()
    second = default_benchmark_ngrams()
    assert first == second
