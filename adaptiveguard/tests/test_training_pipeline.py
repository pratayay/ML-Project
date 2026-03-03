"""Smoke tests for training CLI and artifact outputs."""

from __future__ import annotations

import json
from pathlib import Path

from adaptiveguard.training.train import TrainingConfig, parse_args, persist_artifacts, train_model


def _write_dataset(path: Path) -> None:
    rows = [
        {"comment_text": "kind and helpful message", "toxicity": 0.1, "category": "safe"},
        {"comment_text": "you are awful", "toxicity": 0.9, "category": "abuse"},
        {"comment_text": "neutral statement", "toxicity": 0.2, "category": "safe"},
        {"comment_text": "threatening and hateful words", "toxicity": 0.95, "category": "abuse"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parse_args_smoke() -> None:
    args = parse_args(
        [
            "--data-path",
            "dataset.jsonl",
            "--dataset",
            "jigsaw",
            "--epochs",
            "2",
            "--category-loss-weight",
            "0.2",
        ]
    )

    assert args.data_path == "dataset.jsonl"
    assert args.dataset == "jigsaw"
    assert args.epochs == 2
    assert args.category_loss_weight == 0.2


def test_artifact_paths_are_generated_and_non_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "dataset.jsonl"
    _write_dataset(data_path)

    config = TrainingConfig(
        data_path=data_path,
        dataset="jigsaw",
        output_dir=tmp_path / "outputs",
        epochs=2,
        batch_size=2,
        learning_rate=0.05,
        category_loss_weight=0.2,
        train_split=0.75,
        model_version="test-1",
        seed=7,
    )

    result = train_model(config)
    artifacts = persist_artifacts(result, config)

    assert artifacts
    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0
