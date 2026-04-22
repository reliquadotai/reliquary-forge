"""Tests for :mod:`reliquary.eval.math_harness`.

Covers the pure-Python parts of the eval cycle — the model + tokenizer
are replaced with lightweight stubs so the tests don't pull the 3B
Qwen checkpoint. What remains exercises:

- holdout → prompt → completion → scoring → EvalBundle pipeline
- signed-envelope roundtrip through sign_envelope(EVAL_BUNDLE_TYPE)
- greedy vs temperature sampling code paths
- publish=False smoke path (local dev without R2)
- aggregate_summary excludes per-problem results from log surfaces

Run a full end-to-end test against a real model in a separate
hardware-in-the-loop test suite (not this one).
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

# The tests require the soft-imported reliquary-inference dataset. Skip
# cleanly in envs that don't have the `datasets` library.
pytest.importorskip("reliquary_inference.dataset.math_holdout")
pytest.importorskip("torch")

from reliquary.eval.math_harness import (  # noqa: E402
    EVAL_SYSTEM_PROMPT,
    aggregate_summary,
    render_eval_prompt,
    run_eval_cycle,
)
from reliquary_inference.dataset import math_holdout  # noqa: E402
from reliquary_protocol import (  # noqa: E402
    EVAL_BUNDLE_TYPE,
    HmacBridgeSigner,
    HmacBridgeVerifier,
    verify_envelope,
)


# ---------------------------------------------------------------------------
# Stub model + tokenizer — no HF download
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal HF-compatible tokenizer stub.

    Only needs: ``pad_token_id``, ``eos_token_id``, ``apply_chat_template``,
    ``__call__`` (for building input_ids), ``decode``.
    """

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False
    ) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "\n".join(parts) + ("\n<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        import torch

        # Token id = codepoint mod 256. Good enough for a stub; all we
        # need is a deterministic tensor shape.
        ids = [ord(c) % 256 for c in text[:256]]
        if not ids:
            ids = [0]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
        }

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        # Decode is driven by a per-test monkey-patched completion
        # dispatcher set on the stub model, not the tokenizer. Return
        # a sentinel for tokens we didn't set up.
        return "<stub-decode-fallback>"


class _StubModel:
    """Minimal HF-compatible model stub.

    Emits per-problem completions via a caller-supplied ``completions``
    dict keyed on the deterministic problem prompt prefix. The stub's
    ``generate`` appends the completion tokens onto the input tensor
    and returns the concatenation — exactly the shape the eval harness
    expects from a real HF ``model.generate``.
    """

    def __init__(self, completions: dict[str, str]) -> None:
        self._completions = completions
        # Track the tokenizer the harness is passing in so ``decode``
        # can be redirected at the stub.
        self._tokenizer: _StubTokenizer | None = None

    def parameters(self):
        import torch

        # Single fake parameter with a CPU device so next(...).device is valid.
        t = torch.zeros(1)
        yield t

    def generate(self, **kwargs):
        import torch

        input_ids = kwargs["input_ids"]
        # Look up completion by the first 32 chars of the decoded prompt.
        # Falls back to an "EMPTY" stub to produce a 0-correctness row
        # for un-registered prompts.
        decoded = "".join(chr(int(i) % 256) for i in input_ids[0].tolist()[:128])
        completion_text = None
        for key, completion in self._completions.items():
            if key in decoded:
                completion_text = completion
                break
        if completion_text is None:
            completion_text = "unmatched"

        # Encode the completion by chars → ids; concatenate to input.
        completion_ids = [ord(c) % 256 for c in completion_text][:256]
        if not completion_ids:
            completion_ids = [0]
        completion_tensor = torch.tensor([completion_ids], dtype=torch.long)

        # Pass@k: produce ``num_return_sequences`` copies.
        n_return = int(kwargs.get("num_return_sequences", 1))
        outputs: list[torch.Tensor] = []
        for _ in range(n_return):
            outputs.append(torch.cat([input_ids[0], completion_tensor[0]], dim=0))
        return torch.stack(outputs)


@pytest.fixture
def stubbed_holdout(monkeypatch):
    """Replace the full 500-problem holdout with a tiny 4-problem one
    so the test runs in milliseconds. Mirrors the shape of the real
    load_holdout_problems return value."""

    tiny_problems = [
        {
            "prompt": "What is 1 + 1?",
            "reference_answer": "2",
            "problem_id": hashlib.sha256(b"What is 1 + 1?").hexdigest()[:16],
            "dataset_index": 10,
            "level": "Level 1",
            "subject": "Algebra",
        },
        {
            "prompt": "What is 2 + 2?",
            "reference_answer": "4",
            "problem_id": hashlib.sha256(b"What is 2 + 2?").hexdigest()[:16],
            "dataset_index": 20,
            "level": "Level 1",
            "subject": "Algebra",
        },
        {
            "prompt": "What is 5 * 5?",
            "reference_answer": "25",
            "problem_id": hashlib.sha256(b"What is 5 * 5?").hexdigest()[:16],
            "dataset_index": 30,
            "level": "Level 2",
            "subject": "Geometry",
        },
        {
            "prompt": "What is 10 / 2?",
            "reference_answer": "5",
            "problem_id": hashlib.sha256(b"What is 10 / 2?").hexdigest()[:16],
            "dataset_index": 40,
            "level": "Level 2",
            "subject": "Geometry",
        },
    ]

    def fake_load(**kwargs):
        return list(tiny_problems)

    monkeypatch.setattr(math_holdout, "load_holdout_problems", fake_load)
    return tiny_problems


@pytest.fixture
def tokenizer():
    t = _StubTokenizer()
    # Monkey-patch decode to return the completion we actually produced —
    # the stub's generate appends the completion onto input_ids, so when
    # decode receives (seq[prompt_length:], skip_special_tokens=True),
    # we round-trip back through the char encoding.

    def decode_with_chars(ids, *, skip_special_tokens: bool = True) -> str:
        # ``ids`` is a 1-D tensor of ints. Map back via the stub's
        # "char mod 256" encoding.
        return "".join(chr(int(i) % 256) for i in ids.tolist() if int(i) != 0) or ""

    t.decode = decode_with_chars  # type: ignore[method-assign]
    return t


def _signer_and_verifier():
    signer = HmacBridgeSigner(
        signer_id="test-forge-authority", secret="test-secret-0123"
    )
    verifier = HmacBridgeVerifier(
        secrets={"test-forge-authority": "test-secret-0123"}
    )
    return signer, verifier


# ---------------------------------------------------------------------------
# render_eval_prompt
# ---------------------------------------------------------------------------


def test_render_eval_prompt_includes_system_and_problem(tokenizer):
    rendered = render_eval_prompt("What is 2 + 2?", tokenizer)
    assert "boxed" in EVAL_SYSTEM_PROMPT.lower()
    assert "2 + 2" in rendered


def test_render_eval_prompt_tolerates_none_tokenizer():
    rendered = render_eval_prompt("P", None)
    assert "boxed" in rendered.lower()
    assert "P" in rendered


# ---------------------------------------------------------------------------
# run_eval_cycle — greedy path
# ---------------------------------------------------------------------------


def test_run_eval_cycle_greedy_scores_correctly(stubbed_holdout, tokenizer):
    """Problems 1 + 3 get correct boxed answers; 2 + 4 get wrong ones.
    Expected accuracy = 2/4 = 0.5."""

    # The stub's generate matches prompts by substring. Wrap answers in
    # \boxed{...} so evaluate_math_trace picks them up.
    completions = {
        "1 + 1": r"... \boxed{2}",
        "2 + 2": r"... \boxed{3}",  # wrong — gt is 4
        "5 * 5": r"... \boxed{25}",
        "10 / 2": r"... \boxed{3}",  # wrong — gt is 5
    }
    model = _StubModel(completions)
    signer, _ = _signer_and_verifier()

    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=None,
        signer=signer,
        training_netuid=462,
        inference_netuid=462,
        model_ref="stub://qwen-tiny",
        eval_window_id=12345,
        policy_artifact_id="a" * 64,
        policy_merkle_root_hex="b" * 64,
        max_new_tokens=32,
        sampling_mode="greedy",
        publish=False,
    )
    assert result.bundle.n_problems == 4
    assert result.bundle.accuracy == pytest.approx(0.5)
    assert result.bundle.mean_reward == pytest.approx(0.5)
    # Per-level: Level 1 got 1/2; Level 2 got 1/2.
    assert result.bundle.accuracy_by_level["Level 1"] == pytest.approx(0.5)
    assert result.bundle.accuracy_by_level["Level 2"] == pytest.approx(0.5)
    # Signed envelope verifies.
    _, verifier = _signer_and_verifier()
    assert verify_envelope(result.envelope, verifier) is True
    assert result.envelope.artifact_type == EVAL_BUNDLE_TYPE


def test_run_eval_cycle_empty_holdout_produces_zero_accuracy(monkeypatch, tokenizer):
    monkeypatch.setattr(
        math_holdout, "load_holdout_problems", lambda **kw: []
    )
    signer, _ = _signer_and_verifier()
    model = _StubModel({})
    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=None,
        signer=signer,
        training_netuid=462,
        inference_netuid=462,
        model_ref="stub://qwen-tiny",
        eval_window_id=12345,
        policy_artifact_id="",
        policy_merkle_root_hex="",
        max_new_tokens=32,
        publish=False,
    )
    assert result.bundle.n_problems == 0
    assert result.bundle.accuracy == 0.0


def test_run_eval_cycle_max_problems_cap(stubbed_holdout, tokenizer):
    completions = {
        "1 + 1": r"\boxed{2}",
        "2 + 2": r"\boxed{4}",
        "5 * 5": r"\boxed{25}",
        "10 / 2": r"\boxed{5}",
    }
    model = _StubModel(completions)
    signer, _ = _signer_and_verifier()
    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=None,
        signer=signer,
        training_netuid=462,
        inference_netuid=462,
        model_ref="stub://qwen-tiny",
        eval_window_id=12345,
        policy_artifact_id="",
        policy_merkle_root_hex="",
        max_new_tokens=32,
        max_problems=2,
        publish=False,
    )
    assert result.bundle.n_problems == 2
    # Both picks land in [0, 1] of the tiny holdout — both correct.
    assert result.bundle.accuracy == pytest.approx(1.0)


def test_run_eval_cycle_rejects_unknown_sampling_mode(stubbed_holdout, tokenizer):
    signer, _ = _signer_and_verifier()
    model = _StubModel({"1 + 1": r"\boxed{2}"})
    with pytest.raises(ValueError, match="sampling_mode"):
        run_eval_cycle(
            model=model,
            tokenizer=tokenizer,
            backend=None,
            signer=signer,
            training_netuid=462,
            inference_netuid=462,
            model_ref="stub://x",
            eval_window_id=1,
            policy_artifact_id="",
            policy_merkle_root_hex="",
            sampling_mode="beam",  # not supported
            publish=False,
        )


def test_run_eval_cycle_publish_requires_backend(stubbed_holdout, tokenizer):
    signer, _ = _signer_and_verifier()
    model = _StubModel({"1 + 1": r"\boxed{2}"})
    with pytest.raises(ValueError, match="backend"):
        run_eval_cycle(
            model=model,
            tokenizer=tokenizer,
            backend=None,
            signer=signer,
            training_netuid=462,
            inference_netuid=462,
            model_ref="stub://x",
            eval_window_id=1,
            policy_artifact_id="",
            policy_merkle_root_hex="",
            publish=True,
        )


def test_run_eval_cycle_publishes_to_backend(stubbed_holdout, tokenizer):
    store: dict[str, bytes] = {}

    class _FakeBackend:
        def put(self, key: str, data: bytes) -> None:
            store[key] = data

        def get(self, key: str) -> bytes | None:
            return store.get(key)

    completions = {"1 + 1": r"\boxed{2}", "2 + 2": r"\boxed{4}",
                   "5 * 5": r"\boxed{25}", "10 / 2": r"\boxed{5}"}
    model = _StubModel(completions)
    signer, _ = _signer_and_verifier()
    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=_FakeBackend(),
        signer=signer,
        training_netuid=462,
        inference_netuid=462,
        model_ref="stub://qwen-tiny",
        eval_window_id=12345,
        policy_artifact_id="",
        policy_merkle_root_hex="",
        publish=True,
    )
    assert result.storage_key in store
    assert result.storage_key.startswith("eval_bundles/462/12345/forge-eval-")
    # The put payload must decode via envelope_from_dict round-trip.
    import json

    from reliquary_protocol import envelope_from_dict

    envelope = envelope_from_dict(json.loads(store[result.storage_key]))
    _, verifier = _signer_and_verifier()
    assert verify_envelope(envelope, verifier) is True

    # Rolling index must be updated so web-side discovery can find the bundle.
    index_key = "eval_bundles/462/index.json"
    assert index_key in store
    index_body = json.loads(store[index_key])
    assert index_body["version"] == "1"
    assert index_body["inference_netuid"] == 462
    assert index_body["entries"][0]["eval_run_id"] == result.bundle.eval_run_id
    assert index_body["entries"][0]["accuracy"] == pytest.approx(result.bundle.accuracy)
    assert index_body["entries"][0]["storage_key"] == result.storage_key


def test_update_eval_index_caps_at_max_entries(stubbed_holdout, tokenizer):
    """Growth ceiling — the rolling index must not balloon indefinitely."""
    from reliquary.eval.math_harness import EVAL_INDEX_MAX_ENTRIES, update_eval_index
    from reliquary_protocol import build_eval_bundle

    store: dict[str, bytes] = {}

    class _FakeBackend:
        def put(self, key: str, data: bytes) -> None:
            store[key] = data

        def get(self, key: str) -> bytes | None:
            return store.get(key)

    backend = _FakeBackend()
    for i in range(EVAL_INDEX_MAX_ENTRIES + 20):
        bundle = build_eval_bundle(
            training_netuid=462,
            inference_netuid=462,
            eval_run_id=f"forge-eval-{i:08d}",
            eval_window_id=10_000 + i,
            policy_artifact_id="0" * 64,
            policy_merkle_root_hex="",
            model_ref="stub://t",
            holdout_version="v1",
            holdout_seed="s",
            holdout_dataset="d",
            holdout_size=1,
            holdout_task_ids=["abc"],
            n_samples_per_problem=1,
            sampling_mode="greedy",
            per_problem_results=[{"task_id": "abc", "correctness": 1.0}],
            evaluator_hotkey="eval-a",
        )
        update_eval_index(
            backend=backend, bundle=bundle, storage_key=bundle.storage_key()
        )

    import json

    index_body = json.loads(store["eval_bundles/462/index.json"])
    assert len(index_body["entries"]) == EVAL_INDEX_MAX_ENTRIES
    # Newest-first ordering preserved: the most recent write is slot 0.
    last_id = f"forge-eval-{EVAL_INDEX_MAX_ENTRIES + 19:08d}"
    assert index_body["entries"][0]["eval_run_id"] == last_id


def test_update_eval_index_is_idempotent_on_rerun(stubbed_holdout, tokenizer):
    """Re-publishing the same run_id must not duplicate the entry."""
    from reliquary.eval.math_harness import update_eval_index
    from reliquary_protocol import build_eval_bundle

    store: dict[str, bytes] = {}

    class _FakeBackend:
        def put(self, key: str, data: bytes) -> None:
            store[key] = data

        def get(self, key: str) -> bytes | None:
            return store.get(key)

    backend = _FakeBackend()
    bundle = build_eval_bundle(
        training_netuid=462,
        inference_netuid=462,
        eval_run_id="forge-eval-dup-test",
        eval_window_id=77_777,
        policy_artifact_id="0" * 64,
        policy_merkle_root_hex="",
        model_ref="stub://t",
        holdout_version="v1",
        holdout_seed="s",
        holdout_dataset="d",
        holdout_size=1,
        holdout_task_ids=["abc"],
        n_samples_per_problem=1,
        sampling_mode="greedy",
        per_problem_results=[{"task_id": "abc", "correctness": 1.0}],
        evaluator_hotkey="eval-a",
    )
    update_eval_index(backend=backend, bundle=bundle, storage_key=bundle.storage_key())
    update_eval_index(backend=backend, bundle=bundle, storage_key=bundle.storage_key())
    import json

    index_body = json.loads(store["eval_bundles/462/index.json"])
    ids = [e["eval_run_id"] for e in index_body["entries"]]
    assert ids.count("forge-eval-dup-test") == 1


# ---------------------------------------------------------------------------
# aggregate_summary
# ---------------------------------------------------------------------------


def test_aggregate_summary_excludes_per_problem_results(stubbed_holdout, tokenizer):
    completions = {"1 + 1": r"\boxed{2}", "2 + 2": r"\boxed{4}",
                   "5 * 5": r"\boxed{25}", "10 / 2": r"\boxed{5}"}
    signer, _ = _signer_and_verifier()
    model = _StubModel(completions)
    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=None,
        signer=signer,
        training_netuid=462,
        inference_netuid=462,
        model_ref="stub://x",
        eval_window_id=1,
        policy_artifact_id="",
        policy_merkle_root_hex="",
        publish=False,
    )
    summary = aggregate_summary(result.bundle)
    assert "per_problem_results" not in summary
    assert "holdout_task_ids" not in summary
    assert summary["n_problems"] == 4
    assert summary["accuracy"] == pytest.approx(1.0)
    assert "Level 1" in summary["accuracy_by_level"]
