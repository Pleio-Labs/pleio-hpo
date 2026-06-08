"""The ``pleio-hpo`` command-line interface.

`pleio-hpo [TEXT]` annotates text (or --file / stdin). Subcommands `download`
and `info` are also available. Exit codes: 0 success-with-codes, 1 pipeline
error, 2 usage error, 3 success-but-no-codes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from pleio_hpo import __version__
from pleio_hpo.types import AnnotationResult

EXIT_OK = 0
EXIT_PIPELINE_ERROR = 1
EXIT_USAGE = 2
EXIT_NO_CODES = 3


def _version_line() -> str:
    from pleio_hpo.hpo import bundled_hpo_version

    return f"pleio-hpo {__version__} (HPO {bundled_hpo_version()})"


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(_version_line())
    ctx.exit(EXIT_OK)


class _DefaultGroup(click.Group):
    """A group that dispatches unknown first tokens to the ``annotate`` command,
    so ``pleio-hpo "some text"`` and ``pleio-hpo --format json ...`` work while
    ``pleio-hpo info`` / ``download`` still route to subcommands."""

    default_cmd = "annotate"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and (args[0] in self.commands or args[0] in ("--help", "--version")):
            return super().parse_args(ctx, args)
        return super().parse_args(ctx, [self.default_cmd, *args])


def _format_text(result: AnnotationResult) -> str:
    lines = []
    for c in result.codes:
        flags = f"  [{', '.join(sorted(c.context_flags))}]" if c.context_flags else ""
        lines.append(f"{c.hpo_id:<12}{c.label[:40]:<40}{c.score:>5.2f}{flags}")
    return "\n".join(lines)


@click.group(cls=_DefaultGroup, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_version,
    help="Show version and pinned HPO release, then exit.",
)
def main() -> None:
    """Map free-text clinical descriptions to HPO codes."""


@main.command()
@click.argument("text", required=False)
@click.option("--file", "file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["text", "tsv", "json"]), default="text")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--include-negated", is_flag=True, default=False)
@click.option("--include-family-history", is_flag=True, default=False)
@click.option("--include-uncertain/--no-include-uncertain", default=True)
@click.option("--no-validator", "no_validator", is_flag=True, default=False)
@click.option("--embedding-threshold", type=float, default=0.8)
@click.option("--validator-threshold", type=float, default=0.5)
def annotate(
    text: str | None,
    file: Path | None,
    fmt: str,
    output: Path | None,
    include_negated: bool,
    include_family_history: bool,
    include_uncertain: bool,
    no_validator: bool,
    embedding_threshold: float,
    validator_threshold: float,
) -> None:
    """Annotate TEXT (or --file PATH, or stdin)."""
    if text is not None:
        source_text = text
    elif file is not None:
        source_text = file.read_text()
    else:
        source_text = sys.stdin.read()

    from pleio_hpo import Annotator

    try:
        annotator = Annotator(
            use_validator=not no_validator,
            embedding_threshold=embedding_threshold,
            validator_threshold=validator_threshold,
        )
        result = annotator.annotate(
            source_text,
            include_negated=include_negated,
            include_family_history=include_family_history,
            include_uncertain=include_uncertain,
        )
    except Exception as exc:  # pipeline failure
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_PIPELINE_ERROR)

    if fmt == "json":
        rendered = result.to_json()
    elif fmt == "tsv":
        rendered = result.to_tsv()
    else:
        rendered = _format_text(result)

    if output is not None:
        output.write_text(rendered + "\n")
    else:
        click.echo(rendered)

    sys.exit(EXIT_OK if result.codes else EXIT_NO_CODES)


@main.command()
def info() -> None:
    """Print diagnostic info: version, HPO release, model paths."""
    from pleio_hpo.embed import DEFAULT_MODEL
    from pleio_hpo.embed_match import default_index_path
    from pleio_hpo.hpo import bundled_hpo_version, default_hpo_path
    from pleio_hpo.validator import default_validator_path

    def status(path: Path) -> str:
        return "present" if path.exists() else "MISSING — run `pleio-hpo download`"

    obo = default_hpo_path()
    idx = default_index_path()
    val = default_validator_path()
    click.echo(_version_line())
    click.echo(f"hpo_release:     {bundled_hpo_version()}")
    click.echo(f"hpo_obo:         {obo} ({status(obo)})")
    click.echo(f"embedding_model: {DEFAULT_MODEL}")
    click.echo(f"embedding_index: {idx} ({status(idx)})")
    click.echo(f"validator:       {val} ({status(val)})")
    spacy_status = "present" if _spacy_model_present() else "MISSING — run `pleio-hpo download`"
    click.echo(f"spacy_model:     {SPACY_MODEL} ({spacy_status})")


SPACY_MODEL = "en_core_web_sm"


def _spacy_model_present() -> bool:
    import importlib.util

    return importlib.util.find_spec(SPACY_MODEL) is not None


ASSETS_REPO = os.environ.get("PLEIO_HPO_ASSETS_REPO", "PleioLabs/pleio-hpo-assets")
# Repo-relative paths -> they land under <package>/data/ via local_dir, matching
# default_index_path() and default_validator_path().
_ASSET_FILES = [
    "hpo/index.npz",
    "validator/v1.0/config.json",
    "validator/v1.0/model.safetensors",
    "validator/v1.0/special_tokens_map.json",
    "validator/v1.0/tokenizer.json",
    "validator/v1.0/tokenizer_config.json",
    "validator/v1.0/vocab.txt",
]


@main.command()
def download() -> None:
    """Fetch the runtime assets (embedding index, validator, spaCy model).

    Downloads ~570 MB once into the installed package's ``data/`` directory, plus
    the ``en_core_web_sm`` spaCy model into the environment. The Hub source repo
    can be overridden with ``PLEIO_HPO_ASSETS_REPO``.
    """
    from pleio_hpo.embed_match import default_index_path
    from pleio_hpo.validator import default_validator_path

    hub_assets_present = default_index_path().exists() and default_validator_path().exists()
    if hub_assets_present and _spacy_model_present():
        click.echo("All assets present.")
        return

    if not hub_assets_present:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise click.ClickException(
                "huggingface_hub is required for `download`: pip install huggingface_hub"
            ) from None

        data_root = default_index_path().parent.parent  # <package>/data
        click.echo(f"Downloading assets from {ASSETS_REPO} (~570 MB, one time)...")
        try:
            for name in _ASSET_FILES:
                hf_hub_download(repo_id=ASSETS_REPO, filename=name, local_dir=str(data_root))
        except PermissionError as e:
            raise click.ClickException(
                f"Cannot write assets into {data_root} ({e}). Install pleio-hpo in a "
                "virtualenv (writable), or set PLEIO_HPO_ASSETS_REPO and HF_HOME."
            ) from None

    if not _spacy_model_present():
        click.echo(f"Installing spaCy model {SPACY_MODEL}...")
        import spacy.cli

        spacy.cli.download(SPACY_MODEL)

    click.echo("Done — assets installed.")


if __name__ == "__main__":
    main()
