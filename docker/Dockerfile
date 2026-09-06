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

# Installa PyTorch Geometric e dipendenze
RUN pip3 install torch_geometric
RUN pip3 install --no-build-isolation torch_scatter
RUN pip3 install pfrl

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

# --- BEGIN LIBSIGNAL INTEGRATION ---
# Clona il repository di LibSignal e lo aggiunge al PYTHONPATH
# Viene salvato in /opt/LibSignal così non viene oscurato dal mount di /workspace
RUN git clone https://github.com/DaRL-LibSignal/LibSignal.git /opt/LibSignal
ENV PYTHONPATH="/workspace:/opt/LibSignal:${PYTHONPATH}"
# Installa eventuali dipendenze specifiche non coperte da requirements.txt
RUN pip3 install gym
# Crea un MOCK di SUMO per ingannare LibSignal ed evitare sys.exit
ENV SUMO_HOME="/opt/sumo"
RUN mkdir -p /opt/sumo/tools && \
    touch /opt/sumo/tools/__init__.py && \
    touch /opt/sumo/tools/traci.py && \
    touch /opt/sumo/tools/sumolib.py
# Sostituiamo sys.exit con pass in world_sumo.py nel caso in cui chiami sys.exit altrove
RUN sed -i 's/sys.exit/pass # sys.exit/g' /opt/LibSignal/world/world_sumo.py || true
# LibSignal importa incondizionatamente libsumo. Rimuoviamo questo import per evitare errori.
RUN sed -i 's/import libsumo/pass/g' /opt/LibSignal/world/world_sumo.py || true
# Fix typo di LibSignal (RLAgent_t non definito -> RLAgent)
RUN sed -i 's/RLAgent_t/RLAgent/g' /opt/LibSignal/agent/colight_pytorch_agent.py || true
# --- END LIBSIGNAL INTEGRATION ---

CMD ["/bin/bash"]
