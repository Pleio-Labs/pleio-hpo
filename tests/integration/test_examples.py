"""Run every example script end-to-end and assert it exits cleanly (Task 13.4).

Each example loads the full pipeline, so this is skipped when the embedding
index / validator are not present (same guard as the other integration tests).
The venv's bin dir is prepended to PATH so the shell example finds `pleio-hpo`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pleio_hpo.embed_match import default_index_path
from pleio_hpo.validator import default_validator_path

pytestmark = pytest.mark.skipif(
    not (default_index_path().exists() and default_validator_path().exists()),
    reason="requires bundled embedding index + validator checkpoint (built locally / via download)",
)

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.parametrize(
    "script",
    ["02-library-basics.py", "03-batch-processing.py", "04-context-filtering.py",
     "05-custom-thresholds.py", "01-cli-basics.sh"],
)
def test_example_runs(script: str) -> None:
    path = EXAMPLES / script
    cmd = ["bash", str(path)] if script.endswith(".sh") else [sys.executable, str(path)]
    # The .sh example calls the `pleio-hpo` console script; make sure it's found.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stderr[-2000:]}"
