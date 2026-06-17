FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Python to unbuffered mode for logging
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
