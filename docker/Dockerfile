FROM ubuntu:22.04

LABEL maintainer="MetaSTGAT Implementation"
LABEL description="Docker image for MetaSTGAT traffic signal control"

# Evita prompt interattivi durante apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 -> python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Installa PyTorch (CPU)
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installa PyTorch Geometric
RUN pip3 install torch_geometric

# Installa CityFlow
RUN pip3 install cityflow || (git clone https://github.com/cityflow-project/CityFlow.git /tmp/CityFlow && \
    cd /tmp/CityFlow && pip3 install . && cd /)

# Crea la directory di lavoro
WORKDIR /workspace

# Copia i requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copia il progetto
COPY . .

# Crea cartelle necessarie
RUN mkdir -p results data data/hangzhou data/jinan

CMD ["/bin/bash"]
