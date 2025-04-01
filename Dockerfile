FROM python:3.11-slim

# CUDA 11.8 kurulumu
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2004-11-8-local/7fa2af80.pub /etc/apt/trusted.gpg.d/ \
    && apt-get update \
    && apt-get install -y cuda-11.8.0 \
    && rm -rf /var/lib/apt/lists/*

# Ollama kurulumu
RUN curl https://ollama.ai/install.sh | sh

# curl kurulumu
RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 