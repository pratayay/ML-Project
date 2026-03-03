"""Training CLI and pipeline for AdaptiveGuard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Example:
    text: str
    risk_score: float
    category: str


@dataclass
class TrainingConfig:
    data_path: Path
    dataset: str
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    category_loss_weight: float
    train_split: float
    model_version: str
    seed: int


class SimpleRiskModel:
    """Tiny linear model with an optional category head."""

    def __init__(self, dim: int, categories: list[str], use_category_head: bool) -> None:
        self.dim = dim
        self.categories = categories
        self.w = [0.0 for _ in range(dim)]
        self.b = 0.0
        self.use_category_head = use_category_head
        self.category_w: dict[str, list[float]] = {
            category: [0.0 for _ in range(dim)] for category in categories
        }
        self.category_b: dict[str, float] = {category: 0.0 for category in categories}

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value < -50:
            return 0.0
        if value > 50:
            return 1.0
        return 1 / (1 + math.exp(-value))

    def forward_score(self, x: list[float]) -> float:
        raw = sum(weight * feature for weight, feature in zip(self.w, x, strict=True)) + self.b
        return self._sigmoid(raw)

    def forward_category_logits(self, x: list[float]) -> dict[str, float]:
        return {
            category: sum(weight * feature for weight, feature in zip(self.category_w[category], x, strict=True))
            + self.category_b[category]
            for category in self.categories
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "categories": self.categories,
            "w": self.w,
            "b": self.b,
            "use_category_head": self.use_category_head,
            "category_w": self.category_w,
            "category_b": self.category_b,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaptiveGuard risk model")
    parser.add_argument("--data-path", required=True, help="Path to training dataset")
    parser.add_argument(
        "--dataset",
        choices=["jigsaw", "hatexplain", "civil_comments"],
        required=True,
        help="Dataset adapter to use",
    )
    parser.add_argument("--output-dir", default="artifacts/training", help="Directory for outputs")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Optimizer step size")
    parser.add_argument(
        "--category-loss-weight",
        type=float,
        default=0.0,
        help="Weight for optional category classification loss",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction for training")
    parser.add_argument("--model-version", default="0.1.0", help="Model version string")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    return parser.parse_args(argv)


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_category(value: Any) -> str:
    text = _normalize_text(value)
    return text or "unknown"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON dataset must be a list of objects")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _adapt_jigsaw(rows: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for row in rows:
        text = _normalize_text(row.get("comment_text") or row.get("text"))
        risk = _to_float(row.get("toxicity") or row.get("target") or row.get("risk_score"))
        category = _normalize_category(row.get("category") or row.get("label") or "toxicity")
        if text:
            examples.append(Example(text=text, risk_score=min(max(risk, 0.0), 1.0), category=category))
    return examples


def _adapt_hatexplain(rows: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for row in rows:
        post_tokens = row.get("post_tokens")
        text = " ".join(post_tokens) if isinstance(post_tokens, list) else _normalize_text(row.get("text"))
        label_value = row.get("label")
        category = _normalize_category(label_value or row.get("category") or "hatespeech")
        if isinstance(label_value, str):
            risk = 1.0 if label_value.lower() in {"hate", "hatespeech", "offensive"} else 0.0
        else:
            risk = _to_float(row.get("risk_score") or label_value)
        if text:
            examples.append(Example(text=text, risk_score=min(max(risk, 0.0), 1.0), category=category))
    return examples


def _adapt_civil_comments(rows: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for row in rows:
        text = _normalize_text(row.get("text") or row.get("comment_text"))
        risk = _to_float(row.get("toxicity") or row.get("target") or row.get("risk_score"))
        category = _normalize_category(row.get("category") or ("toxic" if risk >= 0.5 else "non_toxic"))
        if text:
            examples.append(Example(text=text, risk_score=min(max(risk, 0.0), 1.0), category=category))
    return examples


def load_dataset(data_path: Path, dataset: str) -> list[Example]:
    rows = _load_rows(data_path)
    adapters = {
        "jigsaw": _adapt_jigsaw,
        "hatexplain": _adapt_hatexplain,
        "civil_comments": _adapt_civil_comments,
    }
    examples = adapters[dataset](rows)
    if not examples:
        raise ValueError("Dataset adapter returned no valid rows")
    return examples


def to_unified_schema(examples: list[Example]) -> list[dict[str, Any]]:
    return [asdict(example) for example in examples]


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % dim


def vectorize(text: str, dim: int) -> list[float]:
    vector = [0.0 for _ in range(dim)]
    tokens = [token.lower() for token in text.split() if token.strip()]
    for token in tokens:
        vector[_hash_token(token, dim)] += 1.0
    if tokens:
        size = float(len(tokens))
        vector = [value / size for value in vector]
    return vector


def split_dataset(examples: list[Example], train_split: float, seed: int) -> tuple[list[Example], list[Example]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1)")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_split))
    split = min(split, len(shuffled) - 1)
    return shuffled[:split], shuffled[split:]


def make_dataloader(data: list[Example], batch_size: int, shuffle: bool, seed: int):
    idxs = list(range(len(data)))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = [data[i] for i in idxs[start : start + batch_size]]
        yield batch


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def train_model(config: TrainingConfig) -> dict[str, Any]:
    examples = load_dataset(config.data_path, config.dataset)
    unified = to_unified_schema(examples)
    train_data, val_data = split_dataset(examples, config.train_split, config.seed)

    categories = sorted({example.category for example in examples})
    model = SimpleRiskModel(dim=256, categories=categories, use_category_head=config.category_loss_weight > 0)

    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        running_loss = 0.0
        total = 0
        for batch in make_dataloader(train_data, config.batch_size, shuffle=True, seed=config.seed + epoch):
            for sample in batch:
                x = vectorize(sample.text, model.dim)
                pred = model.forward_score(x)
                diff = pred - sample.risk_score
                reg_loss = diff * diff
                grad_scale = 2 * diff * pred * (1 - pred)

                for idx, value in enumerate(x):
                    model.w[idx] -= config.learning_rate * grad_scale * value
                model.b -= config.learning_rate * grad_scale

                cat_loss = 0.0
                if model.use_category_head:
                    logits = model.forward_category_logits(x)
                    probs = _softmax(logits)
                    target = sample.category
                    target_prob = max(probs.get(target, 1e-12), 1e-12)
                    cat_loss = -math.log(target_prob)
                    for category in model.categories:
                        expected = 1.0 if category == target else 0.0
                        error = probs[category] - expected
                        for idx, value in enumerate(x):
                            model.category_w[category][idx] -= (
                                config.learning_rate * config.category_loss_weight * error * value
                            )
                        model.category_b[category] -= config.learning_rate * config.category_loss_weight * error

                running_loss += reg_loss + config.category_loss_weight * cat_loss
                total += 1

        train_loss = running_loss / max(total, 1)
        val_metrics = evaluate(model, val_data)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_mae": val_metrics["mae"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_calibration_error": val_metrics["calibration_error"],
            }
        )

    return {
        "model": model,
        "history": history,
        "unified_count": len(unified),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "categories": categories,
        "final_eval": evaluate(model, val_data),
    }


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranks, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def calibration_error(y_true: list[int], y_score: list[float], bins: int = 10) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    bucket_error = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        members = [idx for idx, score in enumerate(y_score) if lower <= score < upper or (bucket == bins - 1 and score == 1.0)]
        if not members:
            continue
        avg_score = sum(y_score[idx] for idx in members) / len(members)
        avg_true = sum(y_true[idx] for idx in members) / len(members)
        bucket_error += (len(members) / total) * abs(avg_true - avg_score)
    return bucket_error


def evaluate(model: SimpleRiskModel, data: list[Example]) -> dict[str, float]:
    if not data:
        return {"mae": 0.0, "roc_auc": 0.5, "calibration_error": 0.0}
    y_score: list[float] = []
    y_true_reg: list[float] = []
    y_true_bin: list[int] = []
    for sample in data:
        x = vectorize(sample.text, model.dim)
        pred = model.forward_score(x)
        y_score.append(pred)
        y_true_reg.append(sample.risk_score)
        y_true_bin.append(1 if sample.risk_score >= 0.5 else 0)
    mae = sum(abs(pred - true) for pred, true in zip(y_score, y_true_reg, strict=True)) / len(y_score)
    return {
        "mae": mae,
        "roc_auc": roc_auc_score(y_true_bin, y_score),
        "calibration_error": calibration_error(y_true_bin, y_score),
    }


def persist_artifacts(result: dict[str, Any], config: TrainingConfig) -> dict[str, Path]:
    output_dir = config.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    metadata_dir = output_dir / "metadata"
    eval_dir = Path("adaptiveguard/evaluation")

    for path in [output_dir, checkpoints_dir, metrics_dir, metadata_dir, eval_dir]:
        path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"model_{config.model_version}.json"
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(result["model"].to_dict(), handle, indent=2)

    metrics_path = metrics_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result["history"], handle, indent=2)

    metadata_payload = {
        "model_version": config.model_version,
        "dataset": config.dataset,
        "source_data": str(config.data_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "category_loss_weight": config.category_loss_weight,
        "train_size": result["train_size"],
        "val_size": result["val_size"],
        "categories": result["categories"],
        "unified_schema_rows": result["unified_count"],
    }
    metadata_path = metadata_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    eval_path = eval_dir / "evaluation_metrics.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(result["final_eval"], handle, indent=2)

    return {
        "checkpoint": checkpoint_path,
        "metrics": metrics_path,
        "metadata": metadata_path,
        "evaluation": eval_path,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrainingConfig(
        data_path=Path(args.data_path),
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        category_loss_weight=args.category_loss_weight,
        train_split=args.train_split,
        model_version=args.model_version,
        seed=args.seed,
    )

    result = train_model(config)
    artifacts = persist_artifacts(result, config)
    print(f"Training complete. Artifacts: {artifacts}")


if __name__ == "__main__":
    main()
