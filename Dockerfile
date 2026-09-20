FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

# 避免安裝時互動式問答卡住
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-pip \
    python3.12-dev \
    pipx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN pipx install poetry
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction

COPY src/ ./src/

CMD ["python3", "src/droneImageAnalysis/pipeline.py"]