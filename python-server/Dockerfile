# Use Python 3.9 base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    g++ \
    gcc \
    wget \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Manually install NumPy 1.x to avoid compatibility issues
RUN pip install --no-cache-dir "numpy<2"

# Copy the Python requirements file
COPY requirements.txt ./

# Install dependencies with pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Set environment variables to disable OpenBLAS OpenMP issue
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Copy Python script
COPY compliance_api.py ./

# Expose Flask API port
EXPOSE 5000

# Run the Flask application
CMD ["python", "compliance_api.py"]
