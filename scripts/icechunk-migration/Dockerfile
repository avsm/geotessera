# Supply a linux/amd64 Python 3.13 slim image by digest (see README).
ARG PYTHON_IMAGE=python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285
FROM ${PYTHON_IMAGE}
ARG PYTHON_IMAGE
RUN python -c 'import re, sys; assert re.search(r"@sha256:[0-9a-f]{64}$", sys.argv[1]), "Pin PYTHON_IMAGE by digest"' "$PYTHON_IMAGE"
RUN apt-get update -qq && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY scripts/icechunk-migration/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /app/requirements.lock
COPY pyproject.toml README.md LICENSE.md /app/
COPY geotessera /app/geotessera
RUN python -m pip install --no-deps --no-build-isolation . && mkdir /scratch
CMD ["geotessera-registry", "zarr-transcode", "--help"]
