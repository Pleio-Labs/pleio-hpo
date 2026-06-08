# Container image for pleio-hpo.
#
# Base image pinned by digest (not just the :3.11-slim tag) so rebuilds are
# byte-stable. Refresh the digest deliberately with:
#   docker pull python:3.11-slim && \
#   docker inspect --format '{{index .RepoDigests 0}}' python:3.11-slim
FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /work

# Install dependencies first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]" \
    && python -m spacy download en_core_web_sm

COPY . .

# The model assets (validator + embedding index, ~570 MB) are NOT baked into the
# image — fetch them on demand at run time, either by:
#   (a) running `pleio-hpo download` inside the container (needs network), or
#   (b) mounting a host dir with prebuilt assets over the package data dir:
#       -v "$PWD/src/pleio_hpo/data:/work/src/pleio_hpo/data"
CMD ["bash"]
