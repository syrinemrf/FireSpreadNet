#!/usr/bin/env bash
# Substitute BACKEND_URL into nginx config at container start.
# Cloud Run injects BACKEND_URL as an env var at deploy time.
set -e

TEMPLATE=/etc/nginx/templates/nginx.conf.template
TARGET=/etc/nginx/conf.d/default.conf

# Extract just the hostname for use in Host header and proxy_ssl_name
# e.g. https://firespreadnet-api-xxx.run.app → firespreadnet-api-xxx.run.app
BACKEND_HOST=$(echo "${BACKEND_URL}" | sed 's|https\?://||' | cut -d'/' -f1)

sed \
  -e "s|BACKEND_URL_PLACEHOLDER|${BACKEND_URL}|g" \
  -e "s|BACKEND_HOST_PLACEHOLDER|${BACKEND_HOST}|g" \
  "$TEMPLATE" > "$TARGET"

echo "[entrypoint] Backend URL:  ${BACKEND_URL}"
echo "[entrypoint] Backend Host: ${BACKEND_HOST}"

# Validate config before starting
nginx -t

# Hand off to nginx
exec nginx -g "daemon off;"
