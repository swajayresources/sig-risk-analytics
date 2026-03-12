# Production Deployment Guide

## 🎯 Overview

This guide provides comprehensive instructions for deploying the Quantitative Risk Analytics Engine in production environments, including cloud platforms, on-premises infrastructure, and hybrid deployments.

## 📋 Prerequisites

### Infrastructure Requirements

#### Minimum Production Setup
- **Compute**: 8 vCPUs, 32GB RAM
- **Storage**: 500GB SSD (1TB+ recommended)
- **Network**: 10Gbps network interface
- **OS**: Ubuntu 20.04 LTS or RHEL 8+

#### Recommended Production Setup
- **Compute**: 16+ vCPUs, 64GB+ RAM
- **Storage**: 2TB+ NVMe SSD with 10K+ IOPS
- **Network**: 25Gbps+ with low latency
- **OS**: Ubuntu 22.04 LTS
- **Accelerators**: NVIDIA GPU for Monte Carlo (optional)

### Software Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
sudo pip3 install docker-compose

# Install monitoring tools
sudo apt install -y htop iotop nethogs sysstat
```

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended for Single Server)

#### Production Docker Compose Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
 risk-engine:
 image: risk-analytics:latest
 deploy:
 resources:
 limits:
 cpus: '8'
 memory: 16G
 reservations:
 cpus: '4'
 memory: 8G
 environment:
 - ENVIRONMENT=production
 - LOG_LEVEL=WARNING
 - WORKERS=8
 - MAX_CONNECTIONS=5000
 volumes:
 - /opt/risk-engine/data:/app/data
 - /opt/risk-engine/logs:/app/logs
 - /opt/risk-engine/config:/app/config
 ports:
 - "8000:8000"
 restart: unless-stopped
 healthcheck:
 test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
 interval: 30s
 timeout: 10s
 retries: 3

 # Database cluster setup
 postgres:
 image: postgres:15-alpine
 deploy:
 resources:
 limits:
 cpus: '4'
 memory: 8G
 environment:
 POSTGRES_DB: risk_analytics
 POSTGRES_USER: risk_user
 POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
 volumes:
 - postgres_data:/var/lib/postgresql/data
 secrets:
 - postgres_password
 restart: unless-stopped

 redis-cluster:
 image: redis:7-alpine
 command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
 deploy:
 replicas: 6
 resources:
 limits:
 cpus: '2'
 memory: 4G
 restart: unless-stopped

secrets:
 postgres_password:
 external: true

volumes:
 postgres_data:
 driver: local
```

#### Deployment Commands

```bash
# Create secrets
echo "your_secure_password" | docker secret create postgres_password -

# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f
```

### Option 2: Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
 name: risk-engine
 labels:
 name: risk-engine

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
 name: risk-config
 namespace: risk-engine
data:
 risk_config.yaml: |
 # Production risk configuration
 position_engine:
 max_positions: 1000000
 monitoring_interval_ms: 100

 risk_calculation:
 monte_carlo:
 default_scenarios: 100000
 num_threads: 0 # Auto-detect

 performance:
 computation_targets:
 max_var_calculation_time_ms: 5
```

#### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: risk-engine
 namespace: risk-engine
spec:
 replicas: 3
 selector:
 matchLabels:
 app: risk-engine
 template:
 metadata:
 labels:
 app: risk-engine
 spec:
 containers:
 - name: risk-engine
 image: risk-analytics:latest
 ports:
 - containerPort: 8000
 env:
 - name: ENVIRONMENT
 value: "production"
 - name: LOG_LEVEL
 value: "INFO"
 - name: POSTGRES_URL
 valueFrom:
 secretKeyRef:
 name: database-secrets
 key: postgres-url
 - name: REDIS_URL
 valueFrom:
 secretKeyRef:
 name: database-secrets
 key: redis-url
 resources:
 requests:
 cpu: "2"
 memory: "4Gi"
 limits:
 cpu: "8"
 memory: "16Gi"
 volumeMounts:
 - name: config-volume
 mountPath: /app/config
 - name: data-volume
 mountPath: /app/data
 livenessProbe:
 httpGet:
 path: /health
 port: 8000
 initialDelaySeconds: 30
 periodSeconds: 10
 readinessProbe:
 httpGet:
 path: /health
 port: 8000
 initialDelaySeconds: 5
 periodSeconds: 5
 volumes:
 - name: config-volume
 configMap:
 name: risk-config
 - name: data-volume
 persistentVolumeClaim:
 claimName: risk-data-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
 name: risk-engine-service
 namespace: risk-engine
spec:
 selector:
 app: risk-engine
 ports:
 - protocol: TCP
 port: 80
 targetPort: 8000
 type: LoadBalancer

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
 name: risk-engine-ingress
 namespace: risk-engine
 annotations:
 nginx.ingress.kubernetes.io/rewrite-target: /
 cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
 tls:
 - hosts:
 - risk.yourdomain.com
 secretName: risk-engine-tls
 rules:
 - host: risk.yourdomain.com
 http:
 paths:
 - path: /
 pathType: Prefix
 backend:
 service:
 name: risk-engine-service
 port:
 number: 80
```

#### Database Setup

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
 name: postgres
 namespace: risk-engine
spec:
 serviceName: postgres
 replicas: 1
 selector:
 matchLabels:
 app: postgres
 template:
 metadata:
 labels:
 app: postgres
 spec:
 containers:
 - name: postgres
 image: postgres:15
 env:
 - name: POSTGRES_DB
 value: "risk_analytics"
 - name: POSTGRES_USER
 value: "risk_user"
 - name: POSTGRES_PASSWORD
 valueFrom:
 secretKeyRef:
 name: database-secrets
 key: postgres-password
 ports:
 - containerPort: 5432
 volumeMounts:
 - name: postgres-storage
 mountPath: /var/lib/postgresql/data
 resources:
 requests:
 cpu: "1"
 memory: "2Gi"
 limits:
 cpu: "4"
 memory: "8Gi"
 volumeClaimTemplates:
 - metadata:
 name: postgres-storage
 spec:
 accessModes: ["ReadWriteOnce"]
 resources:
 requests:
 storage: 100Gi
 storageClassName: ssd
```

#### Deployment Commands

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets
kubectl create secret generic database-secrets \
 --from-literal=postgres-password=your_secure_password \
 --from-literal=postgres-url=postgresql://risk_user:your_secure_password@postgres:5432/risk_analytics \
 --from-literal=redis-url=redis://redis:6379 \
 -n risk-engine

# Deploy all components
kubectl apply -f k8s/

# Verify deployment
kubectl get all -n risk-engine
kubectl logs -f deployment/risk-engine -n risk-engine
```

### Option 3: Cloud Platform Deployments

#### AWS Deployment with EKS

```bash
# Create EKS cluster
eksctl create cluster \
 --name risk-analytics \
 --version 1.24 \
 --region us-east-1 \
 --nodegroup-name standard-workers \
 --node-type m5.2xlarge \
 --nodes 3 \
 --nodes-min 1 \
 --nodes-max 10 \
 --managed

# Install ALB Ingress Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Deploy using Helm
helm repo add risk-engine https://charts.risk-engine.com
helm install risk-engine risk-engine/risk-analytics \
 --namespace risk-engine \
 --create-namespace \
 --set ingress.enabled=true \
 --set ingress.annotations."kubernetes\.io/ingress\.class"=alb
```

#### Azure Deployment with AKS

```bash
# Create resource group
az group create --name rg-risk-analytics --location eastus

# Create AKS cluster
az aks create \
 --resource-group rg-risk-analytics \
 --name aks-risk-analytics \
 --node-count 3 \
 --node-vm-size Standard_D8s_v3 \
 --enable-addons monitoring \
 --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group rg-risk-analytics --name aks-risk-analytics

# Deploy application
kubectl apply -f k8s/
```

#### Google Cloud Deployment with GKE

```bash
# Create GKE cluster
gcloud container clusters create risk-analytics \
 --machine-type=n1-standard-8 \
 --num-nodes=3 \
 --zone=us-central1-a \
 --enable-autoscaling \
 --min-nodes=1 \
 --max-nodes=10

# Get credentials
gcloud container clusters get-credentials risk-analytics --zone=us-central1-a

# Deploy with Cloud SQL
kubectl apply -f k8s/gcp/
```

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificates with Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d risk.yourdomain.com

# Configure nginx with SSL
cat > /etc/nginx/sites-available/risk-engine << EOF
server {
 listen 443 ssl http2;
 server_name risk.yourdomain.com;

 ssl_certificate /etc/letsencrypt/live/risk.yourdomain.com/fullchain.pem;
 ssl_certificate_key /etc/letsencrypt/live/risk.yourdomain.com/privkey.pem;

 ssl_protocols TLSv1.2 TLSv1.3;
 ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
 ssl_prefer_server_ciphers off;

 location / {
 proxy_pass http://localhost:8000;
 proxy_set_header Host \$host;
 proxy_set_header X-Real-IP \$remote_addr;
 proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
 proxy_set_header X-Forwarded-Proto \$scheme;
 }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/risk-engine /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow 22/tcp # SSH
sudo ufw allow 80/tcp # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw allow 8000/tcp # Risk Engine API

# Database access (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 5432 # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379 # Redis
sudo ufw allow from 10.0.0.0/8 to any port 8123 # ClickHouse
```

### Authentication Setup

```bash
# Generate JWT secret
openssl rand -hex 32 > /opt/risk-engine/secrets/jwt_secret

# Create API keys
python3 -c "
import secrets
import hashlib
api_key = secrets.token_urlsafe(32)
api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
print(f'API Key: {api_key}')
print(f'Hash: {api_key_hash}')
" >> /opt/risk-engine/secrets/api_keys.txt
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
 scrape_interval: 15s

scrape_configs:
 - job_name: 'risk-engine'
 static_configs:
 - targets: ['risk-engine:8000']
 metrics_path: '/metrics'
 scrape_interval: 5s

 - job_name: 'node-exporter'
 static_configs:
 - targets: ['node-exporter:9100']

 - job_name: 'postgres'
 static_configs:
 - targets: ['postgres-exporter:9187']

 - job_name: 'redis'
 static_configs:
 - targets: ['redis-exporter:9121']

rule_files:
 - "alert_rules.yml"

alerting:
 alertmanagers:
 - static_configs:
 - targets:
 - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: risk-engine-alerts
 rules:
 - alert: HighLatency
 expr: risk_engine_request_duration_seconds{quantile="0.95"} > 0.01
 for: 2m
 labels:
 severity: warning
 annotations:
 summary: "High request latency detected"
 description: "95th percentile latency is {{ $value }}s"

 - alert: VaRCalculationFailed
 expr: risk_engine_var_calculation_failures_total > 0
 for: 1m
 labels:
 severity: critical
 annotations:
 summary: "VaR calculation failures detected"
 description: "{{ $value }} VaR calculations failed in the last minute"

 - alert: HighMemoryUsage
 expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
 for: 5m
 labels:
 severity: warning
 annotations:
 summary: "High memory usage"
 description: "Memory usage is above 90%"

 - alert: DatabaseConnectionLoss
 expr: risk_engine_database_connections_active == 0
 for: 1m
 labels:
 severity: critical
 annotations:
 summary: "Database connection lost"
 description: "No active database connections"
```

### Grafana Dashboards

```json
{
 "dashboard": {
 "id": null,
 "title": "Risk Engine Dashboard",
 "panels": [
 {
 "title": "Request Rate",
 "type": "graph",
 "targets": [
 {
 "expr": "rate(risk_engine_requests_total[5m])",
 "legendFormat": "{{method}} {{endpoint}}"
 }
 ]
 },
 {
 "title": "VaR Calculation Time",
 "type": "graph",
 "targets": [
 {
 "expr": "risk_engine_var_calculation_duration_seconds",
 "legendFormat": "VaR Calculation Time"
 }
 ]
 },
 {
 "title": "Portfolio Value",
 "type": "singlestat",
 "targets": [
 {
 "expr": "risk_engine_portfolio_value_total",
 "legendFormat": "Portfolio Value"
 }
 ]
 }
 ]
 }
}
```

## 🔧 Performance Tuning

### Operating System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 1000000" >> /etc/security/limits.conf
echo "* hard nofile 1000000" >> /etc/security/limits.conf

# Optimize kernel parameters
cat >> /etc/sysctl.conf << EOF
# Network optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Memory optimizations
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# CPU optimizations
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0
EOF

sysctl -p
```

### Database Optimization

#### PostgreSQL Tuning

```sql
-- postgresql.conf optimizations
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();

-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_positions_symbol_time
 ON positions (symbol, last_update DESC);

CREATE INDEX CONCURRENTLY idx_trades_timestamp
 ON trades (timestamp DESC)
 WHERE timestamp > NOW() - INTERVAL '30 days';
```

#### Redis Optimization

```bash
# redis.conf optimizations
echo "maxmemory 16gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
echo "save 900 1" >> /etc/redis/redis.conf
echo "save 300 10" >> /etc/redis/redis.conf
echo "save 60 10000" >> /etc/redis/redis.conf
echo "tcp-keepalive 60" >> /etc/redis/redis.conf
echo "timeout 300" >> /etc/redis/redis.conf

systemctl restart redis
```

### Application Performance

#### Intel MKL Configuration

```bash
# Optimize Intel MKL for your hardware
export MKL_NUM_THREADS=8
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=8,MKL_FFT=4"
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

#### NUMA Optimization

```bash
# Check NUMA topology
numactl --hardware

# Bind process to specific NUMA node
numactl --cpunodebind=0 --membind=0./risk_engine

# Configure Docker with NUMA awareness
docker run --cpuset-cpus="0-7" --cpuset-mems="0" risk-analytics:latest
```

## 📈 Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# /etc/nginx/nginx.conf
upstream risk_engine_backend {
 least_conn;
 server 10.0.1.10:8000 weight=3;
 server 10.0.1.11:8000 weight=3;
 server 10.0.1.12:8000 weight=3;
 keepalive 32;
}

server {
 listen 80;
 location / {
 proxy_pass http://risk_engine_backend;
 proxy_http_version 1.1;
 proxy_set_header Connection "";
 proxy_set_header Host $host;
 proxy_set_header X-Real-IP $remote_addr;
 proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
 }
}
```

#### Auto-scaling with Kubernetes

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
 name: risk-engine-hpa
 namespace: risk-engine
spec:
 scaleTargetRef:
 apiVersion: apps/v1
 kind: Deployment
 name: risk-engine
 minReplicas: 3
 maxReplicas: 20
 metrics:
 - type: Resource
 resource:
 name: cpu
 target:
 type: Utilization
 averageUtilization: 70
 - type: Resource
 resource:
 name: memory
 target:
 type: Utilization
 averageUtilization: 80
 - type: Pods
 pods:
 metric:
 name: risk_engine_requests_per_second
 target:
 type: AverageValue
 averageValue: "100"
```

### Database Scaling

#### PostgreSQL Read Replicas

```bash
# Create read replica
pg_basebackup -h master-db -D /var/lib/postgresql/replica -U replication -P -W

# Configure replica
echo "standby_mode = 'on'" >> /var/lib/postgresql/replica/recovery.conf
echo "primary_conninfo = 'host=master-db port=5432 user=replication'" >> /var/lib/postgresql/replica/recovery.conf

# Start replica
sudo -u postgres pg_ctl start -D /var/lib/postgresql/replica
```

#### Redis Cluster Setup

```bash
# Create Redis cluster
redis-cli --cluster create \
 10.0.1.10:7000 10.0.1.11:7000 10.0.1.12:7000 \
 10.0.1.10:7001 10.0.1.11:7001 10.0.1.12:7001 \
 --cluster-replicas 1
```

## 🔄 Backup and Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump -h localhost -U risk_user risk_analytics | gzip > "$BACKUP_DIR/postgres_$DATE.sql.gz"

# Redis backup
redis-cli --rdb "$BACKUP_DIR/redis_$DATE.rdb"

# ClickHouse backup
clickhouse-client --query "BACKUP DATABASE risk_analytics TO Disk('backups', 'clickhouse_$DATE.zip')"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.zip" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR" s3://risk-engine-backups/$(date +%Y/%m/%d)/ --recursive
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh - Disaster recovery script

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
 echo "Usage: $0 <backup_file>"
 exit 1
fi

# Stop services
docker-compose down

# Restore PostgreSQL
gunzip -c "$BACKUP_FILE" | psql -h localhost -U risk_user risk_analytics

# Restore Redis
redis-cli --rdb "$BACKUP_FILE"

# Restart services
docker-compose up -d

echo "Recovery completed from $BACKUP_FILE"
```

## 🚨 Troubleshooting

### Common Production Issues

#### High CPU Usage

```bash
# Identify CPU-intensive processes
top -p $(pgrep -d',' risk_engine)

# Check for runaway calculations
grep "calculation_timeout" /var/log/risk-engine/error.log

# Monitor thread usage
ps -eLf | grep risk_engine | wc -l
```

#### Memory Leaks

```bash
# Monitor memory usage over time
while true; do
 ps -p $(pgrep risk_engine) -o pid,vsz,rss,comm --no-headers
 sleep 60
done > memory_usage.log

# Check for memory fragmentation
cat /proc/buddyinfo

# Force garbage collection (if applicable)
kill -USR1 $(pgrep risk_engine)
```

#### Database Performance Issues

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename,
 pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check indexes
SELECT schemaname, tablename, indexname,
 pg_size_pretty(pg_relation_size(indexname)) as size
FROM pg_indexes
ORDER BY pg_relation_size(indexname) DESC;
```

### Emergency Procedures

#### Service Recovery

```bash
#!/bin/bash
# emergency_restart.sh

echo "Emergency service restart initiated at $(date)"

# Save current state
docker-compose exec risk-engine python -c "
from risk_engine import PositionEngine
engine = PositionEngine()
positions = engine.get_all_positions()
print(f'Current positions: {len(positions)}')
"

# Graceful shutdown with timeout
timeout 30s docker-compose stop risk-engine || docker-compose kill risk-engine

# Clean restart
docker-compose up -d risk-engine

# Verify service health
sleep 10
curl -f http://localhost:8000/health || echo "Health check failed"
```

#### Data Corruption Recovery

```bash
#!/bin/bash
# data_recovery.sh

# Stop all services
docker-compose down

# Check database integrity
docker run --rm -v postgres_data:/var/lib/postgresql/data postgres:15 \
 pg_ctl start && pg_dump --data-only risk_analytics > integrity_check.sql

# Restore from latest backup if corruption detected
if [ $? -ne 0 ]; then
 echo "Database corruption detected, restoring from backup..."./restore.sh /opt/backups/latest_postgres.sql.gz
fi

# Restart services
docker-compose up -d
```

## 📊 Maintenance Procedures

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh - Run weekly

# Update system packages
apt update && apt upgrade -y

# Clean Docker resources
docker system prune -f
docker volume prune -f

# Rotate logs
logrotate /etc/logrotate.d/risk-engine

# Update SSL certificates
certbot renew --quiet

# Database maintenance
psql -c "VACUUM ANALYZE;" risk_analytics
psql -c "REINDEX DATABASE risk_analytics;" risk_analytics

# Check disk space
df -h | awk '$5 > 80 {print "WARNING: " $1 " is " $5 " full"}'

# Monitor service health
curl -f http://localhost:8000/health || echo "Service health check failed"
```

### Performance Monitoring

```bash
#!/bin/bash
# performance_check.sh - Run hourly

# Log key metrics
echo "$(date): Performance check" >> /var/log/performance.log

# CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU Usage: $CPU_USAGE%" >> /var/log/performance.log

# Memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')
echo "Memory Usage: $MEM_USAGE" >> /var/log/performance.log

# Disk I/O
DISK_UTIL=$(iostat -x 1 1 | tail -n +4 | awk '{print $10}' | sort -n | tail -1)
echo "Max Disk Utilization: $DISK_UTIL%" >> /var/log/performance.log

# Network throughput
NET_RX=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
echo "Network RX bytes: $NET_RX" >> /var/log/performance.log

# Application metrics
API_REQUESTS=$(curl -s http://localhost:8000/metrics | grep risk_engine_requests_total | tail -1)
echo "API Requests: $API_REQUESTS" >> /var/log/performance.log
```

---

This deployment guide provides comprehensive instructions for production deployment. For questions or issues, refer to the main README.md or create an issue in the repository.