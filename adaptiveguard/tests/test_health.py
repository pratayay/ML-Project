"""Baseline tests for API health endpoint."""

from adaptiveguard.api.main import health


def test_app_import_and_startup_contract() -> None:
    assert health() == {"status": "ok"}
