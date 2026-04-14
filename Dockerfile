FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/

RUN pip install --no-cache-dir --upgrade pip uv

COPY . /app

RUN uv pip install --system --no-cache -e .

ENV RELIQUARY_DATA_DIR=/app/data
ENV RELIQUARY_DEVICE=cpu
ENV RELIQUARY_MODEL_NAME=sshleifer/tiny-gpt2

CMD ["reliquary", "--help"]
