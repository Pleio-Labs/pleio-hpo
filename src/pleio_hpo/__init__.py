"""pleio-hpo: offline HPO code extraction from free-text clinical notes.

Public API:
    Annotator         -- the main entry point; maps text to HPO codes
    AnnotationResult  -- the result object returned by Annotator.annotate
    Code              -- a single returned HPO code

Everything else under ``pleio_hpo`` is internal and not part of the stable
v1.0 surface.
"""

from importlib.metadata import PackageNotFoundError, version

from pleio_hpo.annotator import Annotator
from pleio_hpo.types import AnnotationResult, Code

try:
    __version__ = version("pleio-hpo")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["Annotator", "AnnotationResult", "Code", "__version__"]
