from pathlib import Path

from reliquary.utils.modeling import _resolve_pretrained_ref


def test_resolve_pretrained_ref_marks_existing_paths_local(tmp_path: Path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    resolved_ref, kwargs = _resolve_pretrained_ref(str(model_dir))
    assert resolved_ref == str(model_dir.resolve())
    assert kwargs == {"local_files_only": True}


def test_resolve_pretrained_ref_preserves_repo_ids() -> None:
    resolved_ref, kwargs = _resolve_pretrained_ref("Qwen/Qwen3-4B")
    assert resolved_ref == "Qwen/Qwen3-4B"
    assert kwargs == {}
