#!/bin/bash
# deploy.sh — One-command setup for MED-RAG on a fresh EC2 instance
# Domain: medrag.online
# Usage: bash deploy.sh
set -euo pipefail

DOMAIN="medrag.online"
EMAIL="${CERTBOT_EMAIL:-your-email@gmail.com}"

echo "========================================="
echo "  MED-RAG — AWS EC2 Deployment Setup"
echo "  Domain: $DOMAIN"
echo "========================================="

# ---- 1. System Updates ----
echo "[1/8] Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y

# ---- 2. Install Docker ----
echo "[2/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    if command -v yum &> /dev/null; then
        sudo yum install -y docker
    else
        sudo apt-get install -y docker.io
    fi
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "Docker installed."
else
    echo "Docker already installed."
fi

# ---- 3. Install Docker Compose ----
echo "[3/8] Installing Docker Compose..."
if ! docker compose version &> /dev/null 2>&1; then
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
    sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed."
else
    echo "Docker Compose already available."
fi

# ---- 4. Install Node.js ----
echo "[4/8] Installing Node.js..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash - 2>/dev/null || \
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo yum install -y nodejs 2>/dev/null || sudo apt-get install -y nodejs
    echo "Node.js installed."
else
    echo "Node.js already installed."
fi

# ---- 5. Build Frontend ----
echo "[5/8] Building frontend..."
cd frontend
npm install
npm run build
cd ..
echo "Frontend built to frontend/dist/"

# ---- 6. Start services with HTTP-only nginx (needed to get SSL cert) ----
echo "[6/8] Starting services (HTTP-only for initial cert setup)..."
cp nginx.init.conf nginx.active.conf

# Temporarily use the HTTP-only config
sudo docker-compose -f docker-compose.prod.yml up -d --build

echo "Waiting 15s for services to stabilize..."
sleep 15

# ---- 7. Obtain SSL Certificate ----
echo "[7/8] Obtaining SSL certificate from Let's Encrypt..."
sudo docker-compose -f docker-compose.prod.yml run --rm --entrypoint certbot certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    --keep-until-expiring \
    -d "$DOMAIN" \
    -d "www.$DOMAIN"

# ---- 8. Switch to HTTPS nginx config and reload ----
echo "[8/8] Enabling HTTPS..."
# nginx.conf already has the full HTTPS config
sudo docker-compose -f docker-compose.prod.yml restart frontend

echo ""
echo "========================================="
echo "  ✅ MED-RAG Deployed Successfully!"
echo "========================================="
echo ""
echo "  🌐 Website:  https://$DOMAIN"
echo "  🔧 Backend:  https://$DOMAIN/chat/message"
echo ""
echo "  Logs:      sudo docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop:      sudo docker-compose -f docker-compose.prod.yml down"
echo "  Restart:   sudo docker-compose -f docker-compose.prod.yml restart"
echo ""
echo "  ⚠️  Don't forget to:"
echo "     1. Point $DOMAIN DNS A-record to this server's IP"
echo "     2. Ingest your medical datasets (python scripts/ingest_data.py)"
echo ""
