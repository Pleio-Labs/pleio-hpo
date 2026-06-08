"""Smoke test: the package imports and exposes its public surface."""

from __future__ import annotations

import logging

import pleio_hpo
from pleio_hpo import Annotator


def test_version() -> None:
    # __version__ is read from the installed package metadata, not hard-coded.
    from importlib.metadata import version

    assert pleio_hpo.__version__ == version("pleio-hpo")
    assert pleio_hpo.__version__[0].isdigit()


def test_public_surface() -> None:
    assert set(pleio_hpo.__all__) == {"Annotator", "AnnotationResult", "Code", "__version__"}
    assert hasattr(pleio_hpo, "Annotator")
    assert hasattr(pleio_hpo, "AnnotationResult")
    assert hasattr(pleio_hpo, "Code")


def test_log_level_default_is_warning(monkeypatch) -> None:
    # Annotator() construction is lazy (no asset load), so this is a pure-unit check.
    monkeypatch.delenv("PLEIO_HPO_LOG_LEVEL", raising=False)
    Annotator()
    assert logging.getLogger("pleio_hpo").level == logging.WARNING


def test_log_level_env_var_honored(monkeypatch) -> None:
    monkeypatch.setenv("PLEIO_HPO_LOG_LEVEL", "DEBUG")
    Annotator()
    assert logging.getLogger("pleio_hpo").level == logging.DEBUG


def test_log_level_explicit_arg_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("PLEIO_HPO_LOG_LEVEL", "DEBUG")
    Annotator(log_level="ERROR")
    assert logging.getLogger("pleio_hpo").level == logging.ERROR
