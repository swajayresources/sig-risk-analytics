# Multi-stage build for optimal production image
FROM ubuntu:22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Intel MKL (optional, for performance)
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mkl-devel && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build C++ components
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
          .. && \
    make -j$(nproc) && \
    make install

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Production stage
FROM ubuntu:22.04 as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-thread1.74.0 \
    libhdf5-103 \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Intel MKL runtime
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mkl && \
    rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r riskapp && useradd -r -g riskapp riskapp

# Set working directory
WORKDIR /app

# Copy built application and Python dependencies
COPY --from=builder /app/build/risk_engine /usr/local/bin/
COPY --from=builder /app/build/lib* /usr/local/lib/
COPY --from=builder /app/python/ ./python/
COPY --from=builder /app/web/backend/ ./web/backend/
COPY --from=builder /app/config/ ./config/
COPY --from=builder /app/requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data logs && \
    chown -R riskapp:riskapp /app

# Configure environment
ENV PYTHONPATH="/app/python:/app/web/backend:$PYTHONPATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH"
ENV MKL_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER riskapp

# Expose ports
EXPOSE 8000

# Start script
COPY --chown=riskapp:riskapp scripts/start.sh ./start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]

# Development stage (optional)
FROM builder as development

# Install development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    cppcheck \
    doxygen \
    graphviz \
    jupyter-notebook \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development packages
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-benchmark \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    matplotlib \
    seaborn \
    plotly

WORKDIR /app

# Development entry point
CMD ["bash"]