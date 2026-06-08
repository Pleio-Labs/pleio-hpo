.PHONY: install test test-full lint typecheck clean

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

test:
	pytest --cov=src/pleio_hpo

# Full suite incl. integration tests + 90% coverage gate. Requires the model assets
# (run `pleio-hpo download` first); without them the integration tests skip and
# coverage is ~70%. This is what to run before a release.
test-full:
	pytest --cov=src/pleio_hpo --cov-fail-under=90

lint:
	ruff check src/ tests/

typecheck:
	mypy src/pleio_hpo/

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
