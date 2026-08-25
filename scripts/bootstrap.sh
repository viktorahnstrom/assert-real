#!/usr/bin/env bash
# bootstrap.sh — first-time VPS setup for assert-real
#
# Usage:
#   ssh root@<server> 'bash -s' < scripts/bootstrap.sh
#
# Idempotent — safe to re-run. Tested on Ubuntu 22.04/24.04.
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-deploy}"
SWAP_SIZE="${SWAP_SIZE:-4G}"

echo "=== assert-real VPS bootstrap ==="

# ── Deploy user ──────────────────────────────────────────────────────────────
if ! id "$DEPLOY_USER" &>/dev/null; then
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
    usermod -aG sudo "$DEPLOY_USER"
    echo "$DEPLOY_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$DEPLOY_USER"
    echo "✓ Created user $DEPLOY_USER"
else
    echo "· User $DEPLOY_USER already exists"
fi

# Copy root's authorized_keys to deploy user
if [ -f /root/.ssh/authorized_keys ]; then
    mkdir -p "/home/$DEPLOY_USER/.ssh"
    cp /root/.ssh/authorized_keys "/home/$DEPLOY_USER/.ssh/authorized_keys"
    chown -R "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
    chmod 700 "/home/$DEPLOY_USER/.ssh"
    chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
    echo "✓ Copied SSH keys to $DEPLOY_USER"
fi

# ── SSH hardening ────────────────────────────────────────────────────────────
SSHD_CONFIG="/etc/ssh/sshd_config"
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONFIG"
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
systemctl reload sshd || systemctl reload ssh || true
echo "✓ SSH hardened (root login disabled, password auth disabled)"

# ── Firewall ─────────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq ufw > /dev/null

ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP (Caddy redirect)
ufw allow 443/tcp   # HTTPS
ufw allow 443/udp   # HTTP/3 (QUIC)
echo "y" | ufw enable
echo "✓ Firewall configured"

# ── Swap ─────────────────────────────────────────────────────────────────────
if ! swapon --show | grep -q /swapfile; then
    fallocate -l "$SWAP_SIZE" /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "✓ ${SWAP_SIZE} swap created"
else
    echo "· Swap already active"
fi

# ── Docker ───────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "$DEPLOY_USER"
    systemctl enable --now docker
    echo "✓ Docker installed"
else
    echo "· Docker already installed"
fi

# ── App directory ────────────────────────────────────────────────────────────
APP_DIR="/opt/assert-real"
mkdir -p "$APP_DIR"
chown "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR"
echo "✓ App directory ready at $APP_DIR"

echo ""
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. scp .env $DEPLOY_USER@<server>:$APP_DIR/.env"
echo "  2. scp docker-compose.prod.yml Caddyfile $DEPLOY_USER@<server>:$APP_DIR/"
echo "  3. ssh $DEPLOY_USER@<server> 'cd $APP_DIR && docker compose up -d'"
