FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
ARG VITE_BACKEND_URL=http://localhost:8000
ARG VITE_API_KEY
ARG VITE_MAX_UPLOAD_MB=10
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL VITE_API_KEY=$VITE_API_KEY VITE_MAX_UPLOAD_MB=$VITE_MAX_UPLOAD_MB
RUN npm run build
FROM nginxinc/nginx-unprivileged:1.28-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
