# ── Build stage ──────────────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first (layer-cache friendly)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy source and build
COPY . .
RUN npm run build

# ── Production stage ───────────────────────────────────────────
FROM nginx:1.27-alpine

# Install bash (needed by envsubst script)
RUN apk add --no-cache bash

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config template (with BACKEND_URL_PLACEHOLDER)
COPY nginx.conf /etc/nginx/templates/nginx.conf.template

# Startup script: replaces BACKEND_URL_PLACEHOLDER then launches nginx
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Default backend URL (overridden at Cloud Run deploy time via --env-vars)
ENV BACKEND_URL=http://backend:8000

EXPOSE 8080

CMD ["/docker-entrypoint.sh"]

