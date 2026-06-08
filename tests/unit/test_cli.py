"""Unit tests for cli.py via Click's CliRunner.

The annotation tests run the full pipeline through the CLI.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from pleio_hpo.cli import main
from pleio_hpo.embed_match import default_index_path
from pleio_hpo.validator import default_validator_path

# Skip only the tests that actually run the pipeline; --version and bad-path
# usage errors short-circuit before any model load, so they always run.
needs_artifacts = pytest.mark.skipif(
    not (default_index_path().exists() and default_validator_path().exists()),
    reason="requires bundled embedding index + validator checkpoint (built locally / via download)",
)


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "pleio-hpo" in result.output
    assert "HPO" in result.output


def test_missing_file_is_usage_error() -> None:
    result = CliRunner().invoke(main, ["--file", "definitely_nonexistent_file.txt"])
    assert result.exit_code == 2


@needs_artifacts
def test_annotate_text_contains_macrocephaly() -> None:
    result = CliRunner().invoke(main, ["Patient has macrocephaly."])
    assert result.exit_code == 0
    assert "HP:0000256" in result.output


@needs_artifacts
def test_format_json_is_valid() -> None:
    result = CliRunner().invoke(main, ["--format", "json", "Patient has macrocephaly."])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["codes"]


@needs_artifacts
def test_no_codes_exits_three() -> None:
    result = CliRunner().invoke(main, ["The weather today is pleasant and sunny."])
    assert result.exit_code == 3
