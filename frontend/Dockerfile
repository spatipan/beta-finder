# ── Build stage ──────────────────────────────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Copy nginx config into build stage so the prod stage can pull it via --from
COPY infra/nginx.conf /nginx.conf

# ── Production stage (nginx) ──────────────────────────────────────────────────────
FROM nginx:alpine AS prod

COPY --from=build /app/dist /usr/share/nginx/html

# Nginx config: SPA fallback + proxy /api → backend
COPY --from=build /nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
