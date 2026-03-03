"""Risk model placeholders."""


class RiskModel:
    """Minimal placeholder risk model class."""

    def __init__(self, model_version: str = "0.0.1") -> None:
        self.model_version = model_version

    def predict(self, features: dict) -> float:
        """Return a placeholder risk score until model logic is implemented."""
        _ = features
        return 0.5
