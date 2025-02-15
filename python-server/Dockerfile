# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set non-interactive mode to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3-venv \
    libgl1 \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Point `python` -> python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 3. Upgrade pip / setuptools / wheel
RUN python -m pip install --upgrade pip setuptools wheel

# 4. Install pinned NumPy < 2.0
RUN pip install "numpy<2.0"

# 5. Install PyTorch (GPU version for CUDA 12.1)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Copy and install your requirements
COPY requirements.txt ./
# Make sure you do NOT have `thinc==8.1.0` in this file anymore.
RUN pip install --no-cache-dir -r requirements.txt

# 7. Download spaCy model
RUN python -m spacy download en_core_web_sm

# 8. (Optional) Limit CPU threads for linear algebra to avoid OpenMP conflicts
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# 9. Copy your Python script
COPY compliance_api.py ./

# 10. Expose Flask API port
EXPOSE 5000

# 11. Run the Flask application
CMD ["python", "compliance_api.py"]
