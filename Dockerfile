# syntax=docker/dockerfile:1
# Small, non-root image for evaluation and scenario generation.
#   docker build -t fanet-defense:0.1.0 .                      # numpy-only (evaluate, generate)
#   docker build --build-arg EXTRAS=train -t fanet-defense:train .   # + CPU PyTorch (training)
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ARG EXTRAS=""
RUN python -m venv /opt/venv \
 && if [ -n "$EXTRAS" ]; then \
      /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[${EXTRAS}]"; \
    else \
      /opt/venv/bin/pip install .; \
    fi

FROM python:${PYTHON_VERSION}-slim
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp
USER 10001
WORKDIR /work
ENTRYPOINT ["fanet-defense"]
CMD ["--help"]
