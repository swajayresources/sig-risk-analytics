#!/bin/bash

# Quantitative Risk Analytics Engine Startup Script
set -e

echo "Starting Quantitative Risk Analytics Engine..."

# Environment setup
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export CONFIG_FILE=${CONFIG_FILE:-config/risk_config.yaml}

# Create necessary directories
mkdir -p data logs

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service_name to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "$service_name is ready!"
            return 0
        fi

        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "ERROR: $service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Parse database URLs from environment
if [[ -n "$REDIS_URL" ]]; then
    REDIS_HOST=$(echo $REDIS_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    REDIS_PORT=$(echo $REDIS_URL | sed -n 's/.*:\([0-9]*\).*/\1/p')
    if [[ -n "$REDIS_HOST" && -n "$REDIS_PORT" ]]; then
        wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"
    fi
fi

if [[ -n "$POSTGRES_URL" ]]; then
    POSTGRES_HOST=$(echo $POSTGRES_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    POSTGRES_PORT=$(echo $POSTGRES_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    if [[ -n "$POSTGRES_HOST" && -n "$POSTGRES_PORT" ]]; then
        wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT" "PostgreSQL"
    fi
fi

if [[ -n "$CLICKHOUSE_URL" ]]; then
    CLICKHOUSE_HOST=$(echo $CLICKHOUSE_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    CLICKHOUSE_PORT=$(echo $CLICKHOUSE_URL | sed -n 's/.*:\([0-9]*\).*/\1/p')
    if [[ -n "$CLICKHOUSE_HOST" && -n "$CLICKHOUSE_PORT" ]]; then
        wait_for_service "$CLICKHOUSE_HOST" "$CLICKHOUSE_PORT" "ClickHouse"
    fi
fi

# Database initialization (if needed)
echo "Checking database initialization..."

# Initialize PostgreSQL schema
if [[ -n "$POSTGRES_URL" ]]; then
    python3 -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['POSTGRES_URL'])
    cur = conn.cursor()
    cur.execute('SELECT version();')
    print('PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'PostgreSQL connection failed: {e}')
    exit(1)
"
fi

# Initialize ClickHouse schema
if [[ -n "$CLICKHOUSE_URL" ]]; then
    python3 -c "
import requests
import os
try:
    response = requests.get(f\"{os.environ['CLICKHOUSE_URL']}/ping\")
    if response.status_code == 200:
        print('ClickHouse connection successful')
    else:
        raise Exception(f'HTTP {response.status_code}')
except Exception as e:
    print(f'ClickHouse connection failed: {e}')
    exit(1)
"
fi

# Set up Python path
export PYTHONPATH="/app/python:/app/web/backend:$PYTHONPATH"

# Performance tuning
echo "Configuring performance settings..."

# Set CPU affinity for better NUMA performance
if command -v numactl &> /dev/null; then
    NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
    if [[ "$NUMA_NODES" -gt 1 ]]; then
        echo "NUMA detected, binding to node 0"
        export NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"
    fi
fi

# Intel MKL optimization
if [[ -d "/opt/intel/oneapi/mkl" ]]; then
    source /opt/intel/oneapi/setvars.sh --config="/opt/intel/oneapi/config.txt" --force 2>/dev/null || true
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=${MKL_NUM_THREADS}"
    echo "Intel MKL configured with $MKL_NUM_THREADS threads"
fi

# Memory settings
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072

# Function to handle graceful shutdown
cleanup() {
    echo "Received termination signal, shutting down gracefully..."

    # Kill background processes
    jobs -p | xargs -r kill -TERM

    # Wait for processes to terminate
    wait

    echo "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the main application based on the mode
case "${1:-api}" in
    "api")
        echo "Starting FastAPI web server..."
        cd web/backend
        exec ${NUMA_PREFIX} python3 -m uvicorn main:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers ${WORKERS:-4} \
            --loop uvloop \
            --log-level ${LOG_LEVEL,,} \
            --access-log \
            --no-server-header
        ;;

    "engine")
        echo "Starting C++ Risk Engine..."
        exec ${NUMA_PREFIX} /usr/local/bin/risk_engine
        ;;

    "worker")
        echo "Starting background worker..."
        cd python
        exec ${NUMA_PREFIX} python3 -m risk_engine.worker \
            --config "../$CONFIG_FILE" \
            --log-level ${LOG_LEVEL}
        ;;

    "scheduler")
        echo "Starting task scheduler..."
        cd python
        exec ${NUMA_PREFIX} python3 -m risk_engine.scheduler \
            --config "../$CONFIG_FILE" \
            --log-level ${LOG_LEVEL}
        ;;

    "benchmark")
        echo "Running performance benchmarks..."
        cd build
        exec ${NUMA_PREFIX} ./benchmarks
        ;;

    "test")
        echo "Running test suite..."
        python3 -m pytest python/tests/ -v --cov=risk_engine
        ;;

    "shell")
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;

    *)
        echo "Usage: $0 {api|engine|worker|scheduler|benchmark|test|shell}"
        echo ""
        echo "  api       - Start FastAPI web server (default)"
        echo "  engine    - Start C++ risk calculation engine"
        echo "  worker    - Start background processing worker"
        echo "  scheduler - Start task scheduler"
        echo "  benchmark - Run performance benchmarks"
        echo "  test      - Run test suite"
        echo "  shell     - Start interactive shell"
        exit 1
        ;;
esac